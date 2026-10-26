# Docstring review, 2026-10-26

Docstrings and comments across `pygeoinf2` were reviewed file by file: the
library modules in full, the examples in full, and in the tests the decision
citations only. Planning remarks, development history, comparisons with v1
and citations of decisions were taken out, and each sentence was rewritten
to state its point itself. The review also corrected text that had become
wrong. No code changed: every edited file has the same syntax tree as at
`459bde2` once docstrings are set aside, and comments do not reach it. The
rule this follows is D-10 as revised the same day: nothing in the code cites
a decision.

This file records every point removed, so that nothing worth keeping is
lost. Most need no action, because DECISIONS.md already holds them or they
were history. The ones that do are gathered first. Code problems seen along
the way are in `CODE_ISSUES.md`, not here.

## The tags

| tag | count | meaning | action |
|---|---|---|---|
| `[cite D-nn]` | 186 | a bare citation removed; the docstring now says the point itself | none |
| `[covered D-nn]` | 54 | reasoning or a measurement removed that DECISIONS.md D-nn already records | none |
| `[RECORD]` | 47 | reasoning or a measurement removed that DECISIONS.md does **not** record | place it: a new entry, the appendix, or decide it is not worth keeping |
| `[v1]` | 168 | a comparison with v1 removed; most carry a "Migration note:" | material for a v1-to-v2 migration guide |
| `[history]` | 195 | development narrative, journal section numbers, who decided what | none; git has it |
| `[stale]` | 184 | text that was wrong or out of date, corrected against the code | worth a read, as the record of what was wrong |

834 entries over 86 files. Pure rewording for clarity has no entry.
A few bullets carry two tags, and are counted under the first.

## To place: the [RECORD] entries

Each of these was in a docstring or comment and is not in DECISIONS.md. Most
are measurements behind a default. Several are timings that the appendix
would hold if they are worth keeping; DECISIONS.md says its timings come from
a throttling laptop and should be re-measured before being quoted.

**algebra/spaces.py**

- (at comment above `CoordinateSpace.uses_component_fast_paths`) same comment [RECORD] removed timings not in DECISIONS.md: orthonormalising 50 fields took 1.9 s through `inner_product` against 0.1 s on components (100 transforms); a 30-step Lanczos run took 1047 analyses against 152. The appendix has only the 2650/50 count.
- `ComponentView` [RECORD] removed "which on a sphere were seven transforms per iteration against the operator's two". The motivation for solving on component views is not in DECISIONS.md (D-15 covers component fast paths generally, not the iterative solvers' ComponentView).

**algebra/operators.py**

- `LinearOperator.assembled` [RECORD] removed "Measured, conjugate gradients on a 2000-dimensional operator over a dense Gram matrix: 320 ms stored as Galerkin against 95 ms stored as components." DECISIONS.md does not record why `assembled()` stores the components form, nor this measurement.

**compat.py**

- [RECORD] D-101 says the v1 adapter "is scaffolding ... and it goes with its test when v1 is removed". That plan was never in the docstring and is not added; it stays in DECISIONS.md. Flagged only so the release notes say whether the adapters ship in 2.0.

**numerics/solvers.py**

- `IterativeSolver` [RECORD] removed "On a sphere the vector algebra alone was seven transforms per iteration" (the same figure as removed from `ComponentView` in algebra/spaces.py). DECISIONS.md does not record why the iterative solvers run on a `ComponentView` (D-15 covers component fast paths in general).
- `IterativeSolver.__init__` (rtol) [RECORD] removed the measurement behind the `1e-8` default: on a 40 000-dimensional five-point Laplacian with a 0.01 shift, CG takes 147 / 243 / 301 iterations at rtol 1e-5 / 1e-8 / 1e-10, with relative solution errors 4e-5 / 4e-8 / 4e-10, so 1e-10 costs a quarter more iterations than 1e-8. D-8 records the default but not this measurement, nor that 1e-10 is often unreachable on an ill-conditioned normal operator (a one-line summary of that stays in the docstring).

**numerics/preconditioners.py**

- `BlockPreconditioner` [RECORD] removed the memory measurement: in blocks of 20, four times the dimension costs four times the memory against sixteen for the dense matrix; at dim 2000, 3.5 MB against 97 MB. Not in DECISIONS.md (minor).
- (at comment in `BlockPreconditioner._invert`) same comment [RECORD] removed "cost 11 MB at dim 2000 where the pattern itself is 1.6" (a dict-of-sets pattern against arrays); the one-line reason stays. Minor; not in DECISIONS.md.
- `ColumnThresholdedPreconditioner` [RECORD] removed "Measured at `dim` 2000 with a threshold of 0.1, the whole build peaks at 1.2 MB against 97 MB to form the dense matrix alone". Not in DECISIONS.md (D-47 covers the symmetrisation, not the memory).

**numerics/randomized.py**

- `random_diagonal` (rtol) [RECORD] removed the measurement: on a 120-dimensional operator, the norm-ratio error estimate and the achieved relative error agree within a few per cent from 20 probes (0.042 against 0.045) to 420 (0.0094 against 0.0094). Not in DECISIONS.md (D-51/D-53 cover stochastic estimates generally). The reason for the norm-ratio test stays in brief.

**numerics/functional_calculus.py**

- (at comment in `iter_lanczos_tridiagonalize`) same comment [RECORD] removed the measurement behind single-pass reorthogonalization: on a spectrum from 1e-6 to 1e6 over 200 steps, a run to 250 of 300 dimensions, and tenfold repeated eigenvalues, a second pass moves `max |Q Q^T - I|` from 1.3e-15 to between 1.1e-15 and 1.8e-15. Not in DECISIONS.md. A one-line reason stays in the comment.
- comment above `_DENSE_EIGH_LIMIT` [RECORD] removed the timing table for `eigh_tridiagonal` against `np.linalg.eigh` (k = 3/5/10/20/40: 7.6/8.4/12.5/26.2/65.9 µs against 3.5/4.5/8.6/22.7/80.4 µs, one BLAS thread), which is the basis for the constant 32. Kept: "the crossover is between 20 and 40". Not in DECISIONS.md.
- (at comment above `_DENSE_EIGH_LIMIT`) same comment [RECORD] removed the timings showing a convergence check every k steps is slower (200 probes, `log` at rtol 1e-3: dim 300, 0.037/0.038/0.041 s for checks every 1/2/3 steps; dim 2000, 1.06/1.23/1.37 s; steps 1400/1600/1800). The argument (a skipped check never saves an application) stays in the comment. Not in DECISIONS.md.
- `_eigh_tridiagonal` [RECORD] removed "measured at 27 % of a stochastic log-determinant on a 300-dimensional operator" (the overhead of the convergence check). Not in DECISIONS.md.
- (at `log_determinant` (dense_limit)) same [RECORD] removed the measurements behind the 4000 default: slogdet at 4000 takes 0.8 s on one thread; dense against stochastic on a well-conditioned operator, 4.2x faster at 2000 and 3.6x at 3000; and the reason for 4000 rather than 4096 (this BLAS takes 2.0 s at 4096 against 0.8 s at 4000). D-50 has the threshold but none of these figures.
- comment in `log_determinant`, diagonal branch [RECORD] removed "(measured: 455.4 +/- 2.8 against an exact 456.7 at dimension 1000)" (minor; the stochastic estimate of a diagonal operator's log-determinant).
- comment in `log_determinant`, dense branch [RECORD] removed "at dimension 3000 the factorization of the identity cost 0.37 s, as much as the operator's own, and doubled the memory" (minor).

**numerics/quadratic_forms.py**

- module-level comment on lazy imports [RECORD] removed the import-time measurement: `scipy.stats` and `scipy.optimize` cost 0.26 s of a 0.48 s `import pygeoinf2`, and 0.22 s once both this module and `inference/problem.py` were lazy. Not in DECISIONS.md (the test docstring in tests/test_code_practice.py `TestImportCost` has the 0.32/0.43 s figures).
- `_imhof` [RECORD] removed "it ... exhausted the 800 it was allowed, and warned. Measured at three weights -- the slowest-decaying case -- that cost 106 ms against 3.7 ms here." This is why the integral is a trapezoid rule rather than `scipy.integrate.quad`; not in DECISIONS.md (D-58 covers the method choice, not the quadrature).

**numerics/root_find.py**

- comment before the Brent call [RECORD] removed the measurement "on a chi-squared-like quantity from three starting points: 23-28 solves per root with bisection, 10-13 with Brent". DECISIONS.md D-57 does not record that `monotone_root` uses Brent in `log t` rather than bisection, nor this measurement.

**numerics/optimization.py**

- `Optimizer.__init__` (`rtol:`) [RECORD] removed the measurements behind keeping `rtol=1e-10` tight and adding `ftol` and the precision-loss outcome: on a 60-dimensional quadratic of condition 4.9, L-BFGS stopped at |g| = 1.8e-7 and reported failure after 19 iterations, nonlinear CG at 1.1e-6, steepest descent took 2000 iterations and 5330 evaluations; with `ftol` and the precision outcome the four runs converge in 16 to 25 iterations and 40 to 119 evaluations. DECISIONS.md does not record `ftol`, its no-absolute-floor choice, or the precision-limited success outcome.
- `SteepestDescent._default_line_search` [RECORD] removed the measurement on a quadratic with metric spread 1e3: Wolfe 3376 evaluations against Armijo's 6744, and at 1e2, 417 against 743, with the same iterations and answer. D-54 records the reason but says Armijo "crawled by two orders of magnitude"; this measurement shows about a factor of two. Reconcile the figure in D-54.

**numerics/convex.py**

- `_minimize_on_simplex` [RECORD] removed measurements not in DECISIONS.md. Handing the subproblem to a general nonlinear solver cost fifty seconds per bundle minimization. Over 30 random bundles at 400 iterations, the accelerated method left a smaller KKT residual on 93 per cent, by a median factor of 3300; it ended behind on 2 of 30. Keeping the best iterate took the median from 557 to 3300 and the tenfold share from 87 to 93 per cent. Across the Backus tests the median `Q` condition number was 1e23 and the residual 4.8e-7 at the median, 3.5e-3 at the 99th percentile (the reason `tolerance` and `warn_above` are separate).
- `PrimalKKTSolver` [RECORD] removed the measurement against `ChambollePockSolver` over sixteen directions at noise radius 0.05: fourteen agree to 1e-11, two (where `mu` hit the clip) to 2.4e-3, and v1 behaves identically down to the same multiplier. DECISIONS.md does not record the KKT route's accuracy limit or this comparison.

**geometry/convex.py**

- `Polytope.project` (max_iterations) [RECORD] removed "A thousand rather than the two hundred this used to allow: measured on a ball cut by a half-space in a sixteen-dimensional Sobolev space, two hundred cycles leave the answer 3.7e-4 off the boundary and a thousand reach it exactly". DECISIONS.md does not record the Dykstra cap of 1000 or this measurement.

**probability/gaussian.py**

- `_check_round_trip` [RECORD] removed measurement: CG at `rtol=1e-8` on a covariance of condition 4e10 returned a trace of 217 against a true 289 without complaint. DECISIONS.md has no entry for the round-trip check on the stochastic KL divergence's reference solve.
- `GaussianMeasure.kl_divergence` (dense_limit) [RECORD] removed measurement: at dimension 4000 the dense divergence is exact in 4.4 s, where a hundred probes take 37 s to reach +/- 1.5. D-50 records the 4000 limit and the caution about it, but not these timings.
- `GaussianMeasure.kl_divergence_estimate` (dense_limit) [RECORD] removed measurement and history: the limit "used to be" 512; dense 0.09 / 0.63 / 4.4 s against a hundred probes' 1.4 / 9.3 / 37 s at dimensions 960 / 2000 / 4000. Not in DECISIONS.md or the appendix.
- `GaussianMeasure.kl_divergence_estimate` [RECORD] removed accuracy measurement of the folded stochastic route: on a dense reference of dimension 400 and condition 1e6, 15.14 +/- 0.30 against an exact 15.67, where the old three-estimate route gave -7.0 +/- 22.8. D-63 records the refusal under "auto", not this evidence.
- `GaussianMeasure._kl_similarity` [RECORD] removed the history and measurements behind folding the divergence into one trace of `g(M) = M - 1 - log M`: the old route estimated `tr(C_o^-1 C_s)` and both log-determinants separately, and `KL(mu || mu)` at dimension 289 came out as 369 +/- 35, or -351 +/- 33 with the reference precision to hand. DECISIONS.md has no entry for the single-trace construction or for the solver round-trip check (CG at rtol 1e-8, condition 4e10: trace 217 against 289). The docstring keeps a one-sentence reason.
- `GaussianMeasure.ambient_ball` comment [RECORD] removed timings for the generalized symmetric eigenproblem `(C_gal, G)` against a non-symmetric decomposition of `C_c`: 0.05 s against 0.26 s at dimension 1000, 4.0 s against 15.2 s at 4000. Not in DECISIONS.md.
- `GaussianMeasure._quantile_method_for` [RECORD] removed measurements behind the auto switch from Imhof to the moment-matched chi-square: Imhof 0.8 s at 500 weights and 10.4 s at 2000; matched agreement 9e-5 relative on 500 and 3e-5 on 2000, in 0.001 s. D-58 covers Imhof against the saddlepoint, not this "matched" switch (more than 1000 positive weights, effective rank above 100), which is undocumented in DECISIONS.md. The thresholds are now stated in the docstring.

**plotting/distributions.py**

- `_binned_density_2d` [RECORD] removed accuracy figures: an axis-aligned filter is wrong by 8% of the peak and moves the contour levels by 6% on a correlated cloud; the full kernel is wrong by 1.5% and 0.7%. The binned KDE has no DECISIONS.md entry (D-93 says only "histograms with kernel density").
- `plot_corner` (density) [RECORD] removed timings: binned is about 2000 times faster than `gaussian_kde`, 5.0 s against 2.2 ms per panel at 20 000 draws on a 160 x 160 grid, and 8.9 s of example 26's 12.2 s. Not in DECISIONS.md. The accuracy statement (1.5% and 0.7%) stays, since it informs the choice.

**inference/backus.py**

- `_DualRoute.primal_solver` [RECORD] removed "measured, they agree to 1e-10, which is strong duality checked rather than assumed". DECISIONS.md does not record the measured primal-dual agreement.
- `_DualRoute.support_values` [RECORD] removed "Measured agreement between the three: 1.7e-8 relative between dual and primal, and 2.7e-11 between primal and KKT." Neither D-79 to D-82 nor the appendix records these figures.
- `_LikelihoodRoute._objective` comment [RECORD] removed "which left the fitted misfit 0.8 per cent above the level on one seed in twenty-five". The reason for scaling by the multiplier stays in the comment; the measurement is not in DECISIONS.md.
- `_LikelihoodRoute._fit` comment [RECORD] removed "the optimizer's stopping rules on warm starts are an open point recorded in DECISIONS.md D-80". D-80 records no such point, so the open question (why warm-started Newton-CG stops a step short, and whether the stopping rules should change) is recorded nowhere.

**inference/normal.py**

- `NormalOperator.__init__` [RECORD] removed the measurement "mean difference 0.86 against a mean of norm 2.6 on a random 12-by-8 problem" for the false "noise-free" model-space posterior (`Q^-1 + A* A`, i.e. `R == I`). DECISIONS.md does not record that the model-space formalism refuses an error-free problem, or why.

**symmetric_space/base.py**

- `PreparedPoints` [RECORD] removed the timings that justify preparing points once (measured 2026-08-29): the conversion was 14.5 of 37 ms on a sphere at 1e5 points, 32 of 61 ms on a torus, and 230 of 254 ms (91%) on a bounded box, whose conversion is a Python loop. DECISIONS.md has no entry for `PreparedPoints` or for converting points once at construction.
- `SymmetricSpace.point_evaluation_operator` (nthreads) [RECORD] removed "finufft's default of every core is slower here at every size measured". This measurement fixed the default `nthreads=1`, and DECISIONS.md (D-52, D-89) does not record it.

**symmetric_space/box.py**

- `Box.prepare_points` [RECORD] removed "it was 230 of 254 ms -- 91% -- of an application at 10^5 points (measured 2026-08-29)". Same gap as the `PreparedPoints` item in base.py: DECISIONS.md has nothing on preparing points once.
- `Box.project_function` [RECORD] removed the measurement behind `taper=True`: on the constant one over `[0, 1]` with the default padding, a path integral from 0.05 to 0.95 gives 0.851 against 0.9 (5% error) with a hard cutoff, falling to 7e-6 for a path from 0.4 to 0.6. The taper default is not in DECISIONS.md. Kept a one-line statement of the consequence.

**symmetric_space/sphere.py**

- (at comment above `_TRANSFORM_MIN_POINTS`) same comment [RECORD] removed the measurement behind the thresholds of 200 points and dim 256: measured at lmax 4, 8, 16, 32 and 64 across six point counts, the old thresholds left a factor of 13 at lmax 64 and 1000 points (34 ms direct against 2.6 ms transform). D-89 records the NUFFT route, not the crossover thresholds.
- (at `Sphere.inner_product`) same [RECORD] removed agreement figures not in DECISIONS.md: the quadrature agrees with the component route to 6e-15 at every truncation, radius and sampling tested, and to 2e-14 with one argument a raw pointwise product; when both arguments are off-span grid arrays the two routes were 0.05% to 2% apart. D-87 says the Lebesgue inner product is the grid quadrature, not why it is preferred when both arguments are off-span products (it is the more accurate of the two).
- `Sphere.evaluate` (nthreads) [RECORD] removed the measurement behind `nthreads=1`: on a 16-core machine finufft's default (every core) was 4 to 20 times slower than one thread, 4.4 ms against 21 ms at lmax 128 for every point count from 200 to 10000. D-52 says the NUFFT's `nthreads` is separate from `n_jobs`, but not why it defaults to one.

**backends/mfem.py**

- `_to_mfem` [RECORD] (minor) removed the timing "16 ms for 33k [stored entries]" for the entry-by-entry conversion. Not in DECISIONS.md; probably not worth recording.

## DECISIONS.md entries the review found out of date

These are not docstring points. The review compared docstrings against the
record and the code, and these entries no longer match the code:

- **D-96** says `backends.mfem.operator_from_bilinear_form` "densifies and is
  not the route for a real problem". It no longer densifies: `MfemSpace.restrict`
  keeps CSR and `from_matrix` keeps a sparse input sparse.
- **D-115** gives the extra as `planetmodel[harmonics]>=1.2`. `pyproject.toml`
  requires 1.3, and the batched transform needs 1.3's `grid.kind` (see the
  `sem1d` import-guard entry in `CODE_ISSUES.md`).
- **D-54** says Armijo "crawled by two orders of magnitude on a badly scaled
  metric". The measurement the `SteepestDescent` docstring carried shows about
  a factor of two (3376 evaluations against 6744 at a metric spread of 1e3).
  One of the two needs reconciling.
- **D-101** calls the v1 adapters in `compat` scaffolding, to go with their test
  when v1 is removed. The docstrings now describe the adapters without the
  plan, so the release notes should say whether they ship in 2.0.

## What was not touched

- String literals that are code: error messages, warnings and printed output.
  Two error messages still cite D-131 and D-132, and three tests match on
  "D-131". These, and a few printed lines in the examples, are listed in
  `CODE_ISSUES.md`.
- The regression history in test docstrings, which records why a test
  exists. Only citations and planning references were taken out of the tests.
- The tutorials, which cite nothing; DECISIONS.md, CURRENT_STATE.md and the
  other planning documents (apart from D-10 itself and the opening of
  DECISIONS.md); and the old `pygeoinf` package.
- `compat.py` still talks about v1, since v1 is what it is about.

## The ledger, file by file

### algebra/spaces.py

- module docstring [cite D-14] removed "See DECISIONS.md D-14 to D-24."; added one sentence naming `HilbertModule` as the second capability.
- `HilbertSpace.shares_vectors_with` [history] removed the account of testing identity alone making `from_formal_adjoint` refuse a good lift when a second `EuclideanSpace(n)` was minted (as `from_vectors` and `coordinate_projection` do).
- `CoordinateSpace.white_noise_components` [stale] "both shipped ones do" was vague; now names `DiagonalMetricSpace` and `OrthonormalSpace`.
- `CoordinateSpace.is_function_space` [cite D-131] and [cite D-11] removed; the docstring now names the density-family methods that refuse and points to `AbsolutelyContinuousMeasure` instead of "the sampling layer".
- `CoordinateSpace.squared_norm` [history] removed "a Krylov loop takes a norm every iteration" (motivation only).
- comment above `CoordinateSpace.uses_component_fast_paths` [covered D-15] removed "Coordinate-free when it must be, and not otherwise" and the lmax-64 figure of 2650 analyses for fifty fields (in D-15 and the appendix).
- same comment [RECORD] removed timings not in DECISIONS.md: orthonormalising 50 fields took 1.9 s through `inner_product` against 0.1 s on components (100 transforms); a 30-step Lanczos run took 1047 analyses against 152. The appendix has only the 2650/50 count.
- `CoordinateSpace.white_noise` [cite D-19] removed "the mistake this method exists to avoid; see DECISIONS.md D-19"; it now says standard normal components give covariance `G`, which is `random`.
- `CoordinateSpace.coordinate_inclusion` [cite D-26] removed "the distinction of DECISIONS.md D-26 in its smallest possible setting".
- `CoordinateSpace.coordinate_selection` [v1] removed "v1's `EuclideanSpace.subspace_projection`, on any coordinate space". Migration note: v1 `EuclideanSpace.subspace_projection` becomes `space.coordinate_selection(indices)` on any coordinate space.
- `CoordinateSpace.coordinate_selection` [stale] "so a product through this stays fused" was jargon; reworded.
- `CoordinateSpace.representer` [cite D-26] removed "see DECISIONS.md D-26".
- `HilbertModule` [cite D-87] removed "See DECISIONS.md D-87."
- `MassWeightedSpace` [cite D-85] removed "The construction DECISIONS.md D-85 sets against the Gram matrix".
- `MassWeightedSpace` [stale] removed "Chaining these is how the concrete spaces are built. v1's are exactly such a chain: component-Euclidean, then L2 with a diagonal Gram, then Sobolev through an invariant mass operator on L2." This is false for v2: Sobolev spaces are diagonal-metric spaces, not mass-weighted ones (D-85).
- `MassWeightedSpace` [v1] same sentence. Migration note: v1 built Sobolev spaces as a mass-weighted chain over L2; v2 does not.
- `MassWeightedSpace.__new__` [v1] removed "as v1's did by delegation" and "v1's `MassWeightedHilbertModule`". Migration note: v1 `MassWeightedHilbertModule` is what `MassWeightedSpace(base, M)` returns over a module base.
- `MassWeightedSpace.__init__` [v1] removed "v1 makes the caller supply `inverse_mass_operator` outright; deriving it is one fewer thing to get wrong, and it makes the construction usable when the inverse has no closed form." Migration note: v1 `inverse_mass_operator=` becomes the optional `mass_solver=`, a solver or an inverse operator, CG by default. The rationale is partly in D-22, which does not mention dropping `inverse_mass_operator`.
- `MassWeightedSpace.mass_inverse` comment [history] removed "a conjugate-gradient solve per application was what this used to cost instead".
- `_CoordinateMassWeightedSpace.is_function_space` [cite D-131] removed.
- `_CoordinateMassWeightedModule` [v1] removed "v1's `MassWeightedHilbertModule` over a coordinate base"; the docstring now says what the class is.
- `DiagonalMetricSpace.gram_matrix` [history] removed the dim² multiplications motivation.
- `ComponentView` [RECORD] removed "which on a sphere were seven transforms per iteration against the operator's two". The motivation for solving on component views is not in DECISIONS.md (D-15 covers component fast paths generally, not the iterative solvers' ComponentView).

### algebra/operators.py

- module docstring [cite D-25] removed "See DECISIONS.md D-25 to D-35."
- module docstring [v1] removed "which subsume what v1 calls forms". Migration note: v1 forms are v2 `Functional` / `LinearFunctional`.
- `EvaluationError` [cite D-129] removed "(D-129)". The docstring now also says the optimizer returns the last good point and that a failure at the starting point propagates (both from D-129, checked against `numerics/optimization.py`).
- `LinearOperator.with_traits` [history] removed the account of a wrapper costing a `DiagonalLinearOperator` its exact log-determinant (a hundred Hutchinson probes) and a `NormalOperator` its factors. The reason, dispatch on type, is kept in one sentence.
- `LinearOperator._link_adjoint` [history] removed "rather than writing into `__dict__` by name, which seven call sites in three modules were doing".
- `LinearOperator.adjoints_are_linked` [history] removed "Measured: `solver(A) @ B` cost 60 applications of `A` at dimension 60, while `B @ solver(A)` cost none." The O(n^3) reasoning is kept.
- `LinearOperator.with_probed_adjoint` [covered D-34] removed "v1 derived one silently, by probing every basis vector of the domain on every adjoint application; v2 refuses".
- `LinearOperator.matrix` [cite D-25] removed "See DECISIONS.md D-25."
- `LinearOperator.matrix` row-fill comment [cite D-26] removed "See DECISIONS.md D-26."; the comment now says the row is G_X applied to the components, not the components themselves.
- `LinearOperator.as_scipy` [v1] removed "v1 returned this from `matrix(dense=False)`; here `matrix` means a dense array and this is the other thing." Migration note: v1 `matrix(dense=False)` becomes `as_scipy()`. Covered in substance by D-25.
- `LinearOperator.assembled` [RECORD] removed "Measured, conjugate gradients on a 2000-dimensional operator over a dense Gram matrix: 320 ms stored as Galerkin against 95 ms stored as components." DECISIONS.md does not record why `assembled()` stores the components form, nor this measurement.
- `LinearOperator.assembled` [covered D-88] removed "This is why no observation operator needs a `matrix_free` flag."
- `LinearOperator.assembled` [stale] "converted once, at factorization time, against the `dim` factorization itself" was garbled; reworded to "for less than the factorization itself costs".
- `LinearOperator.from_tensor_product` [cite D-23] removed "see DECISIONS.md D-23"; the docstring now says the library has no tensor product of spaces.
- `LinearOperator.from_tensor_product` [stale] the Args documented `left` and `right`, but the parameters are `u` and `v`; the Returns said `(right, x) left`; *domain* was said to be "taken from the vectors when they carry one", though it is a required keyword and nothing is inferred. All corrected, and the PSD claim when `u is v` is documented.
- `LinearOperator.from_derivative_callables` [cite D-26] removed "the error of DECISIONS.md D-26 in the setting where it is hardest to see". Also fixed "for an operator large to assemble" to "too large", and dropped the Raises text that repeated the paragraph above it.
- `MatrixLinearOperator` [history] removed the account of the matrix constructors returning closures, so that `matrix()`, `diagonals()`, `assembled()` and direct solvers re-derived the matrix ("At dimension 1200 that is 50-70 ms per call and grows as n^3").
- `MatrixLinearOperator` [cite D-25] removed "(see DECISIONS.md D-25)".
- `Functional` [v1] removed "what v1 calls a non-linear form. Subsumed rather than removed". Migration note: v1 `NonLinearForm` is v2 `Functional`. The docstring now lists what the class adds.
- `Functional.from_callables` [cite D-26] removed "the classic error of DECISIONS.md D-26"; it now says what the error is (derivative components passed as a gradient, wrong by a factor of the metric).
- `LinearFunctional` [v1] removed "what v1 calls a `LinearForm`". Migration note: v1 `LinearForm` is v2 `LinearFunctional`; v1 `LinearForm(components=)` maps to `from_derivative_components` or `from_representer` (D-27).
- `LinearFunctional` [cite D-26] removed "See DECISIONS.md D-26."; added "The metric enters at the adjoint and nowhere else."
- `LinearFunctional.from_callables` [history] removed "There was no way to build one of these from a mapping at all: this name resolved by MRO to `LinearOperator.from_callables` ...".
- `LinearFunctional.from_callables` [v1] same passage. Migration note: v1 `LinearForm(domain, mapping=...)` becomes `LinearFunctional.from_callables(domain, value, representer=... | derivative_components=...)`.
- `LinearFunctional.from_callables` [cite D-26] removed "the error of DECISIONS.md D-26".
- `LinearFunctional.from_derivative_components` [stale] the Args documented `components` for the parameter `g`; the docstring said the inverse metric is applied "inside `adjoint`", but it is applied once, at construction (`domain.representer(g)`).
- `AffineOperator` [v1] removed "without the string-based type check v1 uses in `LinearOperator.__add__`". Added a sentence documenting that the translation may be a zero-argument callable.
- `AffineOperator.translation` comment [cite D-40] removed "(DECISIONS.md D-40)".

### algebra/nodes.py

- module docstring [cite D-31] removed "See DECISIONS.md D-31, D-32 and D-36."
- `_Identity._combine_scale` [history] removed "`sigma * I` was a scaled node, which nothing downstream recognizes, so ... stayed a composition and had to be probed a column at a time". The reason is kept, in the present tense.
- `_Composition._known_matrix` comment [covered D-33] removed "at n == 3000 and k == 10 is 72 MB" and "(DECISIONS.md D-33)".
- `_Composition._known_diagonals` comment [stale] "The inner product B" read as an inner product; reworded to "the product of the inner factors".
- section comment above `_FunctionalSum` [history] removed "That mattered concretely. A misfit plus a regularizer is a sum of functionals, and `numerics.optimization` asks for `functional.at(x).gradient` -- so every composed objective ... raised AttributeError."
- same comment [cite D-30] removed "The Hessian rules are DECISIONS.md D-30's"; the comment now states the rules (sum, scaling, `phi @ F` linear or with a second derivative).

### algebra/direct_sum.py

- module docstring [history] removed "the vector wrapper that was rejected everywhere else".
- module docstring [v1] removed "That is not speculative generality: v1 already builds the joint law of model and data as a pushforward of a product measure through a block operator". The construction is kept as a present-tense statement. Migration note: v1 also built the joint prior by block operator; the v2 block operators are nonlinear by default.
- module docstring [cite D-23] removed "See DECISIONS.md D-23."
- `DirectSum.is_function_space` [cite D-131] removed "(D-131)".
- `DirectSum.projection` [stale] "the palindrome rule" was undefined jargon in this module; it now says what is recognized and why it matters.
- `_accumulate` [history] removed "Two allocations that a block operator used to make for nothing ... it now starts from a copy" and "Measured on `[[I, 0], [I, I]]` over two spheres: three syntheses per application before, none after." The behaviour and its reason are kept.

### algebra/diagonal.py

- module docstring [v1] removed "This is the v2 home for what v1 calls `InvariantLinearAutomorphism`". Migration note: v1 `InvariantLinearAutomorphism` is v2 `DiagonalLinearOperator` (D-32 records the correspondence).
- module docstring [cite D-36] removed "That is the rule from DECISIONS.md D-36"; the rule is stated directly. Added that on the symmetric spaces these are the operators invariant under the group action (D-83).
- `DiagonalLinearOperator.__init__` [stale] added the missing `Raises:` section (TypeError, ValueError), which the constructor raises.
- comment above `_rebuild` [cite D-32] removed "The specialization protocol of DECISIONS.md D-32"; it now points to `Operator._combine_add`.
- comment above `_combine_add` [history] "came out claiming nothing at all" rephrased as a present-tense consequence.
- `DiagonalLinearOperator._require_spectrum` [history] removed "as this used to"; the argument against gating on PSD is kept as a counterfactual.

### algebra/linearization.py

- module docstring [cite D-29] removed "See DECISIONS.md D-29."
- comment above `Linearization` [history] the account of `==` and `__hash__` raising was in the past tense; rephrased as the reason for `eq=False`.
- `QuadraticModel` [cite D-26] removed "the classic error described in DECISIONS.md D-26"; it now says what the error is.

### algebra/__init__.py

- no changes needed.

### traits.py

- module docstring [cite D-36] removed "See DECISIONS.md D-36."
- `gramian_traits` [stale] "The palindrome rule" was undefined jargon here; reworded to name the rule.

### parallel.py

- module docstring [cite D-6] removed "The distinction that decides where this is used, from DECISIONS.md D-6 and D-52".
- module docstring [v1] removed "threading a `parallel=` flag down into an operator's own arithmetic is how v1 ended up with the flag on methods that could not use it". Migration note: v1's `parallel=` flags on operator methods become `n_jobs=` on the loops around operators.
- module docstring [v1] removed "as it was in v1" (joblib a required dependency).
- module docstring [covered D-52] removed "Measured on 2026-08-29 (the appendix of `DECISIONS.md`)". The three regimes are kept as user guidance.
- module docstring [covered D-52] removed "measured at 8.6 s against 5.5 s for one dense assembly at four workers".
- module docstring [history] removed "as pyslfp's `run_all.sh` does" (another project's script).
- module docstring [history] "the one value it was there to offer" (no `backend` argument) rephrased without the implied history.
- module docstring [v1] removed "That is the reason v1 keeps its workers in a module of top-level functions"; replaced by the advice itself.

### compat.py

(v1 is this module's subject, so v1 comparisons stay; narrative and citations go.)

- module docstring [stale] removed "The point of this module is to get the existing concrete spaces -- the sphere, circle, torus, plane and line, roughly 9.5k lines of numerically delicate transform code -- usable against the new core immediately, so that the core is exercised on real problems long before those spaces are rewritten natively." That was a plan, and it is now done: the spaces are native in `symmetric_space`. The docstring now describes both halves of the module, the class aliases (which it did not mention) and the adapters.
- module docstring [stale] "importing this module is what pulls in v1" was false: the module never imports v1 (`pygeoinf`); the adapters duck-type what they are given.
- module docstring [cite D-19] removed "See DECISIONS.md D-19."
- `AdaptedSpace.__init__` [stale] cross-reference to a nonexistent `_detect_diagonal_gram` corrected to `_diagonal_gram`; documented the warning on the `auto` dense fallback; added the missing `Raises:` (ValueError for an unknown strategy).
- `adapt_space` [stale] the docstring said `gram="auto"` "uses the space's own Gram if it has one and the identity otherwise". In fact it chooses how white noise factorizes the Gram (probe for diagonal, else dense with a warning).
- `adapt_form` [cite D-26] removed "See DECISIONS.md D-26."
- section comment above the aliases [cite D-99] removed "The inference-layer renames of DECISIONS.md D-99".
- [RECORD] D-101 says the v1 adapter "is scaffolding ... and it goes with its test when v1 is removed". That plan was never in the docstring and is not added; it stays in DECISIONS.md. Flagged only so the release notes say whether the adapters ship in 2.0.

### __init__.py

- module docstring [history] removed "pygeoinf 2.0 -- development tree. This is the shadow package; its design is recorded in `DECISIONS.md` and `CURRENT_STATE.md`. It is not shipped; the released library remains `pygeoinf`." and the side-by-side `import pygeoinf as gi` / `import pygeoinf2 as gi2` note. Replaced by a description of the package and its subpackages, ready for release. **Needs revisiting at the rename**: the docstring describes the package as it will ship.
- module docstring [covered D-102] removed "Imports are relative throughout so that the eventual promotion is a directory rename."
- subpackage-import comment [history] removed "inference and plotting were the two a user needs in order to solve anything, and the two that were not reachable at all; numerics arrived only by accident, pulled in transitively by symmetric_space."
- same comment [cite D-111] removed "See DECISIONS.md D-111"; the comment now names what is not pulled in (compat, backends, datasets, testing; checked against `sys.modules` after `import pygeoinf2`).
- `__all__` comment [v1] removed "as v1 had it" (flat workflow names; D-5).

### datasets.py

- module docstring [cite D-103] removed "(DECISIONS.md D-103)".
- module docstring [v1] removed "v1's `datasets` and `config` modules, with the automatic fetch left out." Migration note: v1 `datasets` and `config` become `pygeoinf2.datasets`; nothing downloads automatically; refresh with `download_gsn_stations()` / `download_usgs_earthquakes()`.
- `download_usgs_earthquakes` [stale] "another name keeps the seed table in use" was unclear ("seed table"); reworded.

### data/__init__.py

- no changes needed.

### numerics/__init__.py

(no changes)

### numerics/solvers.py

- module docstring [cite D-38] removed "See DECISIONS.md D-38."
- module docstring [history] removed "the one piece of v1's solver API that is straightforwardly right, and kept. What changes:"; the bullets now describe the module rather than a difference from v1.
- module docstring [v1] removed "v1 writes `assert operator.is_automorphism`, which vanishes under `python -O` and says nothing about symmetry or definiteness." Migration note: v1's `assert operator.is_automorphism` is replaced by declared `requires` traits and `requires_coordinates`, checked with a `ValueError`.
- module docstring [v1] removed "v1 stores the iteration count on the solver object, so a solver cannot be shared and the count belongs to whichever solve ran last." Migration note: v1's `solver.iterations` (or similar state on the solver) becomes `InverseOperator.solve(y).iterations`, or a `ProgressCallback`.
- module docstring [stale] the list of coordinate-free iterative solvers left out GMRES; added.
- `InverseOperator` [history] removed "now that solvers are stateless".
- `InverseOperator.adjoint_inverse` [history] removed the account of a second private cache that made the operator be inverted twice.
- comment in `InverseOperator._make_adjoint` [stale] "Close the loop, for the same reason the sum and composition nodes do" pointed elsewhere; now states the reason (identity comparison in the expression nodes).
- section headers [stale] the "Direct solvers" header sat above `ProgressCallback`; it now sits above `_Factors`/`DirectSolver`, and the callbacks have a "Callbacks" header.
- `ProgressCallback` [v1] removed "because v1 had it". Migration note: v1 had `ProgressCallback`; v2 keeps it, and it receives a `SolveStep`.
- `SolutionTrackingCallback` [v1] removed "v1's class of the same name". Migration note: v1 `SolutionTrackingCallback` exists in v2 with the same name, and its iterates are in `.iterates`.
- `DirectSolver` [v1] removed "v1 threads a `galerkin=` flag by hand through every call site instead." Migration note: v1's `galerkin=` flag on direct solvers is gone; each solver's `form` class attribute fixes the representation.
- comment in `DirectSolver._invert` [v1] removed "v1 factorized inside every posterior call and so paid the O(n^3) step per use".
- same comment [cite D-40] removed "(DECISIONS.md D-40)"; covered by D-40.
- `DirectSolver._factorize` [history] removed "The first version had a second method for the transposed solve and it factorized again".
- `IterativeSolver` [RECORD] removed "On a sphere the vector algebra alone was seven transforms per iteration" (the same figure as removed from `ComponentView` in algebra/spaces.py). DECISIONS.md does not record why the iterative solvers run on a `ComponentView` (D-15 covers component fast paths in general).
- `IterativeSolver.__init__` (rtol) [RECORD] removed the measurement behind the `1e-8` default: on a 40 000-dimensional five-point Laplacian with a 0.01 shift, CG takes 147 / 243 / 301 iterations at rtol 1e-5 / 1e-8 / 1e-10, with relative solution errors 4e-5 / 4e-8 / 4e-10, so 1e-10 costs a quarter more iterations than 1e-8. D-8 records the default but not this measurement, nor that 1e-10 is often unreachable on an ill-conditioned normal operator (a one-line summary of that stays in the docstring).
- same [v1] removed "1e-5 (v1's default)". Migration note: v1's default iterative rtol was 1e-5; v2's is 1e-8.
- same [history] removed "1e-10 (v2's first choice)" and "That combination is what broke one of the examples during the port."
- `IterativeSolver.with_preconditioner` [history] removed "It used to return the solver unchanged instead", along with the account of a caller's preconditioner being discarded silently. The reason for refusing stays in one clause.
- `IterativeSolver._record` [history] the old run-to-the-cap behaviour was described in the past tense; now conditional ("would run").
- comment in `IterativeSolver._invert` [covered D-41] removed "an exact `P == A^-1` took 13 iterations on `A*` instead of one" and the past-tense account of the generic route.
- `CGSolver` [history] removed "That precondition is now *checked*"; the docstring now also says that non-positive curvature raises `ConvergenceError`.
- `CGSolver._refuse_preconditioner` [stale] the sentence on the cause was garbled; reworded.
- comment in `FlexibleCGSolver._solve` [stale] "the residuals are conjugate" is wrong: with a fixed preconditioner, successive residuals are orthogonal in the preconditioner's inner product, not conjugate.
- `GMRESSolver` [stale] "The only solver here that asks nothing of its operator" was false: `BiCGStabSolver` also requires no traits.
- `LeastSquaresSolver` [v1] removed "In v1 `LSQRSolver` sits under `IterativeLinearSolver`, whose `__call__` asserts the operator is an automorphism". Migration note: v1 `LSQRSolver` was an `IterativeLinearSolver`; in v2 it is a `LeastSquaresSolver`, a sibling of `LinearSolver`.
- `LSQRSolver._solve` [v1] removed "which is what v1 does" (a warm start solves for the correction).

### numerics/preconditioners.py

- module docstring [v1] removed "v1 already models it this way and it is right: nothing else is needed". Added a one-paragraph map of the preconditioners in the module.
- `JacobiPreconditioner` [v1] removed "v1's behavior" (estimating the diagonal from random probes).
- `JacobiPreconditioner.__init__` (samples) [v1] removed "v1 estimated by default, from 20 probes." Migration note: v1's Jacobi preconditioner estimated the diagonal from 20 random probes by default; v2 reads it exactly by default and estimates only with `samples=`.
- `JacobiPreconditioner.__init__` (rtol) [v1] removed "v1's adaptive stopping, which the estimator underneath always had and this did not pass on." Migration note: v1's adaptive diagonal estimation is `rtol=` / `max_samples=` / `block_size=` here.
- `SpectralPreconditioner.__init__` (rank) [v1] removed "v1's `method="variable"`, which the range finder underneath always offered and this did not pass on." Migration note: v1 `method="variable"` is `rank=None` with `rtol=` / `max_rank=` / `block_size=`.
- `_sparse_inverse_traits` [history] removed "both classes used to make it unconditionally ... `check_traits` caught it on a dense Gram"; the condition is now stated directly.
- `BandedPreconditioner` [covered D-46] removed "measured on a random dense positive-definite operator it took conjugate gradients from 125 iterations to failure".
- `BandedPreconditioner.__init__` (form) [stale] the docstring did not say what `"auto"` means; added (Galerkin if the operator claims self-adjointness, else components, as `LinearOperator.diagonals` does).
- `_probe_columns` [v1] removed "v1 probed a matrix-free scipy operator column by column for exactly this reason."
- `sparse_approximation` [v1] removed "This is v1's `with_sparse_approximation` with the measure taken off it". Migration note: v1 `GaussianMeasure.with_sparse_approximation` is `numerics.preconditioners.sparse_approximation(covariance)` (or `GaussianMeasure.sparse_covariance`), which returns an operator, not a measure.
- same [cite D-65] removed "(DECISIONS.md D-65)"; covered.
- same [history] removed "The port had replaced it with a dense assembly, a global threshold and an `O(N^3)` definiteness check."
- `sparse_approximation` (threshold) [v1] removed "and v1's" (the correlation criterion). D-65 records that the correlation criterion is v1's.
- `BlockPreconditioner` [RECORD] removed the memory measurement: in blocks of 20, four times the dimension costs four times the memory against sixteen for the dense matrix; at dim 2000, 3.5 MB against 97 MB. Not in DECISIONS.md (minor).
- `BlockPreconditioner.__init__` (blocks) [history] removed "v2 required a partition"; the reason overlap is allowed stays.
- `BlockPreconditioner.__init__` (form) [stale] now says `"auto"` behaves as for `BandedPreconditioner`.
- comment in `BlockPreconditioner._invert` [v1] removed "That is v1's construction." (overlapping blocks taking entries from the operator, deduplicated).
- same comment [RECORD] removed "cost 11 MB at dim 2000 where the pattern itself is 1.6" (a dict-of-sets pattern against arrays); the one-line reason stays. Minor; not in DECISIONS.md.
- `WoodburyPreconditioner.__init__` (solver) [stale] the default inner solver is also capped at 200 iterations (`CGSolver(rtol=1e-3, max_iterations=200, strict=False)`); the docstring said only rtol and strict.
- `WoodburyPreconditioner.from_normal` [stale] `Raises:` omitted the `ValueError` for a normal operator with no error covariance; added.
- `ColumnThresholdedPreconditioner` [RECORD] removed "Measured at `dim` 2000 with a threshold of 0.1, the whole build peaks at 1.2 MB against 97 MB to form the dense matrix alone". Not in DECISIONS.md (D-47 covers the symmetrisation, not the memory).
- same [history] removed "which is the case it was written for, and the case that forming the dense matrix ruled out".
- same, Note [v1] removed "v1 did not do this." (symmetrising the thresholded pattern). Migration note: v1's column-thresholded preconditioner was not symmetric.
- same, Note [covered D-47] removed the comparison with the untapered truncation of `InvariantDistancePreconditioner` ("cheap to prevent, silent when it happens"); D-47 records the general rule.
- `ColumnThresholdedPreconditioner.__init__` (threshold) [stale] added the fallback in `_keep`: when the diagonal is zero, the column's largest entry is the reference.

### numerics/randomized.py

- module docstring [v1] removed the account of v1 drawing probes from `white_noise_measure`, whose covariance on a mass-weighted space is the Gram matrix, so v1's probes were anisotropic "exactly where `random_range` documents a 'geometric safety guard' for". Migration note: v1's randomized methods drew probes from `white_noise_measure` (covariance `G` on a weighted space); v2 draws from `HilbertSpace.white_noise`, or a measure via `measure=`.
- module docstring [cite D-19] removed "(DECISIONS.md D-19)"; the docstring now states the covariance of the probes directly.
- module docstring [history] removed "The other change is orthogonalization."
- `_probe_range` [v1] removed "which is v1's `measure=`". Migration note: v1's `measure=` on the range finder is kept under the same name.
- `_probe_range` [cite D-53] removed "(DECISIONS.md D-53)"; covered.
- comment in `random_range` [v1] removed "That is v1's arrangement." (rebuilding the whole basis every adaptive round, a cubic cost).
- `LowRankEig` [stale] "the palindrome rule" was undefined jargon; the class now just says self-adjoint by construction and PSD when the eigenvalues are non-negative.
- `LowRankEig.eigenvalues` [stale] "largest first" is wrong: `random_eig` orders by magnitude, so a negative eigenvalue can come first. Now "largest in magnitude first".
- `random_eig` [stale] `Raises:` claimed it refuses a non-endomorphism, which it does not check (self-adjointness implies it), and left out the non-positive rank and the trivial range, both of which raise `ValueError`. Corrected. The adaptive `rank` description now names `rtol`.
- `random_svd` [stale] "no component map is used" was false: on coordinate spaces the arithmetic is done on component arrays. Corrected, and the docstring now says that singular values below 1e-14 of the largest are dropped.
- `random_svd` [stale] `Raises:` said "for a rank exceeding the smaller dimension", but such a rank is silently capped (verified: rank 20 on a 6-dimensional space returns rank 6). Corrected to the conditions that do raise.
- `random_trace` [v1] removed "With v1's probes the expectation is `tr(G A)` instead"; now stated as a property of standard normal component probes.
- `random_trace` (rtol) [v1] removed "v1 grew the sample count until the *estimate* stopped moving, which is a noisy test of a noisy quantity". Migration note: v1's adaptive trace estimate stopped on the change in the estimate; v2 stops on the standard error.
- `random_trace` [stale] `Raises:` said "non-square", but the check is `is_endomorphism`; reworded.
- `deflated_diagonal` (rtol) [stale] described the stopping rule as a "relative change", but `random_diagonal` stops on a standard-error ratio; corrected.
- `deflated_diagonal` [stale] `Raises:` said "non-positive rank or sample count": rank 0 is accepted (it falls back to `random_diagonal`) and no sample count is checked. Corrected to the conditions that do raise (negative rank, a non-self-adjoint operator via `random_eig`, unknown form, tolerance outside (0, 1)).
- `random_diagonal` [cite D-25] removed "(DECISIONS.md D-25)"; the two forms are now named.
- `random_diagonal` (rtol) [RECORD] removed the measurement: on a 120-dimensional operator, the norm-ratio error estimate and the achieved relative error agree within a few per cent from 20 probes (0.042 against 0.045) to 420 (0.0094 against 0.0094). Not in DECISIONS.md (D-51/D-53 cover stochastic estimates generally). The reason for the norm-ratio test stays in brief.

### numerics/functional_calculus.py

- module docstring [cite D-32] removed "That split is the specialization protocol of DECISIONS.md D-32 applied to a unary operation"; the class-versus-trait point stays, stated directly.
- `_guard_spectrum` [stale] "as the dense route already refuses it" did not say which route; it now names the dense `log_determinant`.
- comment in `iter_lanczos_tridiagonalize` [history] removed "The review asked for a second ('twice is enough') pass ... Not done, on the measurement."
- same comment [RECORD] removed the measurement behind single-pass reorthogonalization: on a spectrum from 1e-6 to 1e6 over 200 steps, a run to 250 of 300 dimensions, and tenfold repeated eigenvalues, a second pass moves `max |Q Q^T - I|` from 1.3e-15 to between 1.1e-15 and 1.8e-15. Not in DECISIONS.md. A one-line reason stays in the comment.
- `apply_operator_function` [v1] removed "That is v1's arrangement." (combining the basis once, at the end, after a convergence test on the coefficients).
- `apply_operator_function` [covered D-49] removed "It used to be 50, which the tolerance never met ... a relative error between 1e-3 and 1e-1 (DECISIONS.md D-49)". D-49 records that the cap of 50 never met the tolerance. The detail that a well-conditioned operator needed 11 applications is not recorded (minor).
- `apply_operator_function` (rtol) [covered D-49] removed "`1e-8` here was measured at `1e-6` in the result, `1e-10` at `1e-10`" (the D-49 appendix has rtol 1e-8 delivering 1e-6); replaced by "about two orders of magnitude at 1e-8".
- comment above `_DENSE_EIGH_LIMIT` [RECORD] removed the timing table for `eigh_tridiagonal` against `np.linalg.eigh` (k = 3/5/10/20/40: 7.6/8.4/12.5/26.2/65.9 µs against 3.5/4.5/8.6/22.7/80.4 µs, one BLAS thread), which is the basis for the constant 32. Kept: "the crossover is between 20 and 40". Not in DECISIONS.md.
- same comment [history] removed "The alternative the review proposed ... was prototyped and is a *regression*, so it is not done."
- same comment [RECORD] removed the timings showing a convergence check every k steps is slower (200 probes, `log` at rtol 1e-3: dim 300, 0.037/0.038/0.041 s for checks every 1/2/3 steps; dim 2000, 1.06/1.23/1.37 s; steps 1400/1600/1800). The argument (a skipped check never saves an application) stays in the comment. Not in DECISIONS.md.
- `_eigh_tridiagonal` [RECORD] removed "measured at 27 % of a stochastic log-determinant on a 300-dimensional operator" (the overhead of the convergence check). Not in DECISIONS.md.
- `operator_quadratic_form` [v1] removed "as v1 did" (stopping when the quadrature settles). Migration note: none needed (same behaviour as v1).
- `operator_quadratic_form` [history] removed "It used to run every one of `max_iterations` however early the value converged".
- `operator_function` [history] removed "it always was on the Lanczos route, and the diagonal one used to skip the check"; merged the paragraph "The requirement belongs to *this* function rather than to the calculus" into the first.
- `log_determinant` (dense_limit) [covered D-50] the default 4000 and its role as a memory and factorization budget are in D-50, which stays.
- same [history] removed "The default was 512, which sent a 960-dimensional evidence to a stochastic estimate of +/-30 nats in 4.8 s where the dense route was exact in 2.8 s".
- same [RECORD] removed the measurements behind the 4000 default: slogdet at 4000 takes 0.8 s on one thread; dense against stochastic on a well-conditioned operator, 4.2x faster at 2000 and 3.6x at 3000; and the reason for 4000 rather than 4096 (this BLAS takes 2.0 s at 4096 against 0.8 s at 4000). D-50 has the threshold but none of these figures.
- `log_determinant` [stale] `Raises:` omitted the `ValueError` the dense route raises for a singular or indefinite matrix; added.
- comment in `log_determinant`, diagonal branch [RECORD] removed "(measured: 455.4 +/- 2.8 against an exact 456.7 at dimension 1000)" (minor; the stochastic estimate of a diagonal operator's log-determinant).
- comment in `log_determinant`, dense branch [RECORD] removed "at dimension 3000 the factorization of the identity cost 0.37 s, as much as the operator's own, and doubled the memory" (minor).
- comment in `log_determinant`, dense branch [history] removed "the dense route took an O(dim^3) slogdet of a diagonal matrix".

### numerics/quadratic_forms.py

- module docstring [stale] did not mention the saddlepoint method or the equal-weights shortcut; added one sentence on each.
- module-level comment on lazy imports [RECORD] removed the import-time measurement: `scipy.stats` and `scipy.optimize` cost 0.26 s of a 0.48 s `import pygeoinf2`, and 0.22 s once both this module and `inference/problem.py` were lazy. Not in DECISIONS.md (the test docstring in tests/test_code_practice.py `TestImportCost` has the 0.32/0.43 s figures).
- same comment [history] removed "`scipy.integrate.quad` was imported here too and never used".
- same comment [stale] removed "One other module still imports `scipy.stats` eagerly -- `inference/problem.py` ... until that one follows, this change saves nothing at all". No longer true: `inference/problem.py` now imports `chi2` inside the function, and `import pygeoinf2` leaves `scipy.stats` unloaded (checked).
- `_imhof` [v1] removed "which is v1's" (the vectorized trapezoid rule). Migration note: none needed (same numerics as v1).
- `_imhof` [RECORD] removed "it ... exhausted the 800 it was allowed, and warned. Measured at three weights -- the slowest-decaying case -- that cost 106 ms against 3.7 ms here." This is why the integral is a trapezoid rule rather than `scipy.integrate.quad`; not in DECISIONS.md (D-58 covers the method choice, not the quadrature).
- `weighted_chi2_quantile` (method) [v1] removed "That is v1's behavior too." (the empirical quantile for `monte_carlo`).

### numerics/line_search.py

- module docstring [v1] removed "the slope being metric-correct is precisely what v1's SciPy bridge has to arrange by hand, passing derivative components so that SciPy's Euclidean dot product happens to give the right number". Migration note: v1's optimisers wrapped SciPy and passed derivative components as `jac`; v2's line searches are coordinate-free (D-54 records the v1 behaviour). Added one sentence on how an `EvaluationError` is treated, which the module did not mention.
- `LineSearchResult.model` [history] removed "spent to learn something already known" and "evaluates as before".
- `ArmijoLineSearch.__call__` comment [cite D-129] removed.
- `ArmijoLineSearch.__call__` comment [history] removed "a stop on `min_step` used to be reported as the full backtrack count"; the comment now says what the count is.
- `StrongWolfeLineSearch.__call__` comment [cite D-129] removed.
- `StrongWolfeLineSearch._zoom` [stale] the docstring said bisection is used "when the interpolant puts its minimizer within a tenth of the bracket of either end". The code accepts the cubic only at least a fifth (0.2) of the bracket from either end and the quadratic at a tenth. Corrected.
- `StrongWolfeLineSearch._zoom` [covered D-54] removed "It used to be bisection alone, which converges linearly"; D-54 records the switch to interpolation and its measurements.
- `StrongWolfeLineSearch._zoom` [v1] removed "That is what v1 had through SciPy's search." Migration note: v1 used SciPy's interpolating line search.
- `StrongWolfeLineSearch._zoom` comment [cite D-129] removed.

### numerics/root_find.py

- module docstring [cite D-57] removed "DECISIONS.md D-57 names this as"; the list of users now also names the KKT solvers in `numerics.convex`, which call `monotone_root` too.
- module docstring [history] removed "The solvers have taken an `x0` all along; this is what passes it."
- `_BREAKDOWN` comment [history] removed "though it used to be" (`ConvergenceError` was once read as saturation, reporting "no root in this range" for a sweep whose `maxiter` was too low). The reason it is excluded stays in the comment. DECISIONS.md D-57 does not mention it; the comment is enough.
- `RootResult.breakdown` [history] removed "now".
- `monotone_root` [v1] removed "v1's `minimum_damping`" from `minimum:`; it now says the discrepancy searches expose it as `minimum_damping` (D-57). Migration note: v1 `minimum_damping` survives under that name on the discrepancy searches.
- `monotone_root` [stale] `Raises:` claimed a `ValueError` for "tolerances that cannot close a bracket". No such check exists; the third check is a non-positive `max_iterations`. Corrected. Also moved the paragraph on working in `log t` from after `Raises:` to the description.
- comment before the Brent call [RECORD] removed the measurement "on a chi-squared-like quantity from three starting points: 23-28 solves per root with bisection, 10-13 with Brent". DECISIONS.md D-57 does not record that `monotone_root` uses Brent in `log t` rather than bisection, nor this measurement.
- `DampedSolves.assemble` [history] removed the account of plain sums having refused structure-aware preconditioners in every discrepancy sweep while working at a fixed damping. The reason (`A`, `Q`, `R` are read off the member) stays.

### numerics/quadratic_programming.py

- module docstring [stale] the backends were listed "in the order `best_available_qp_solver` prefers them" as OSQP, Clarabel, SciPy, but the function tries Clarabel first. Reordered, and the descriptions made consistent with the measured preference (Clarabel is called "slower to start" here but faster in D-56).
- module docstring [stale] said the built-in projected gradient "is enough for the proximal method". `ProximalBundleMethod.qp_solver` uses Clarabel or OSQP when installed and falls back to the projected gradient only otherwise (D-56). Corrected.
- module docstring [v1] removed "This module is a port of v1's `convex_optimization.py`". Migration note: v1 `convex_optimization.py` QP backends become `numerics.quadratic_programming`.
- module docstring [covered D-13] removed "whose author's view on the API is to be sought before it is changed beyond the port".
- `OSQPQPSolver` [stale] "the fastest of the three at scale" contradicted D-56's measurement and the backend order; now "operator splitting (ADMM), with warm starts".
- `best_available_qp_solver` [covered D-56] removed "OSQP used to come first" and the measurements: OSQP at its iteration cap on 40 per cent of level-method masters, 0.37 s against 0.89 s, a thousand times closer to the primal. One sentence of the reason stays.

### numerics/optimization.py

- module docstring [v1] removed the account of v1's SciPy wrapper. It passed gradient components where derivative components were wanted ("verified: the ratio ... is exactly the Gram diagonal") beside a Galerkin Hessian, so Newton-CG solved `H p = -G^-1 g`; v1's `line_search` got it right with a comment. D-54 records the substance. The error itself is still described generically. Migration note: v1's SciPy-backed optimisers are replaced by the coordinate-free `SteepestDescent`, `NonlinearCG`, `LBFGS`, `NewtonCG` and `TrustRegionNewton`.
- module docstring [cite D-26] removed "the conditioning half of DECISIONS.md D-26".
- `OptimizationResult.failure` [cite D-129] removed.
- `Optimizer.__init__` (`rtol:`) [RECORD] removed the measurements behind keeping `rtol=1e-10` tight and adding `ftol` and the precision-loss outcome: on a 60-dimensional quadratic of condition 4.9, L-BFGS stopped at |g| = 1.8e-7 and reported failure after 19 iterations, nonlinear CG at 1.1e-6, steepest descent took 2000 iterations and 5330 evaluations; with `ftol` and the precision outcome the four runs converge in 16 to 25 iterations and 40 to 119 evaluations. DECISIONS.md does not record `ftol`, its no-absolute-floor choice, or the precision-limited success outcome.
- `Optimizer.__init__` (`ftol:`) [stale] added that `TrustRegionNewton` does not apply `ftol` (see code issues).
- `Optimizer._precision_limited` [stale] the docstring argued the scale but never stated the test; added `||g|| <= sqrt(eps) (1 + |f|)`.
- `Optimizer._finish_evaluation_failure` [cite D-129] removed.
- `_DescentMethod._minimize` comment [history] removed "which is what turned a converged steepest descent into 2000 iterations" (the figure is in the RECORD item above).
- `SteepestDescent` [cite D-26] removed "That difference is the whole of DECISIONS.md D-26".
- `SteepestDescent._default_line_search` [cite D-54] removed "as DECISIONS.md D-54 says it is" and [history] "It inherited a backtracking one".
- `SteepestDescent._default_line_search` [RECORD] removed the measurement on a quadratic with metric spread 1e3: Wolfe 3376 evaluations against Armijo's 6744, and at 1e2, 417 against 743, with the same iterations and answer. D-54 records the reason but says Armijo "crawled by two orders of magnitude"; this measurement shows about a factor of two. Reconcile the figure in D-54.
- `NewtonCG.__init__` (`forcing:`) [stale] said the inner solve stops at "this fraction of the gradient norm". The code uses `min(forcing, sqrt(||g||))`, which is what makes later iterations accurate. Corrected; `max_cg_iterations:` now says what `None` means.
- `TrustRegionNewton._minimize` comment [cite D-129] removed.
- `truncated_cg` [stale] said both negative curvature and the boundary "return the point where the boundary is crossed". Without a trust region, negative curvature returns the step so far, or `rhs` if none. Corrected; `max_iterations:` now reads "the larger of the dimension and ten", matching `max(space.dim, 10)`.
- `gauss_newton_hessian` [cite D-30] removed "See DECISIONS.md D-30."
- `gauss_newton_hessian` [stale] "positive semidefinite by construction" in `Returns:` holds only when `W` claims PSD or is omitted; a self-adjoint-only `W` gives a self-adjoint-only result (checked). "The palindrome rule" jargon was replaced by a plain statement.
- `gauss_newton_hessian` [stale] `Raises:` claimed a `ValueError` when the operator cannot supply a derivative; that is the operator's `NotImplementedError`. The missing `ValueError` for a weighting not claiming self-adjointness was added.

### numerics/convex.py

- module docstring [stale] "a SciPy-backed quadratic program behind a protocol": the backends are Clarabel, OSQP and SciPy; now points to `quadratic_programming`.
- module docstring [stale] removed "The KKT, Chambolle-Pock and support-value machinery is entangled with it and is deliberately left behind". `ChambollePockSolver`, `PrimalKKTSolver` and `LevelKKTSolver` are in this module (restored under D-13). Replaced with a one-paragraph map of what the module holds.
- module docstring [v1] removed "Ported from v1's `convex_optimization` and `convex_analysis`". Migration note: v1 `convex_optimization` and `convex_analysis` become `numerics.convex` (plus `numerics.quadratic_programming`).
- `SupportFunction` [stale] removed "Only the constructions with no dependence on the inversion layer are brought across"; the list (ball, point, sum, image, scaling) omitted the oracle, half-space and hyperplane constructors. Rewritten to list them all.
- `SupportFunction.of_oracle` [v1] removed "v1's `CallableSupportFunction`". Migration note: v1 `CallableSupportFunction(f, maximizer)` becomes `SupportFunction.of_oracle(domain, f, maximizer=)` or `ConvexSet.from_support_function`.
- `_HalfSpaceSupport` [v1] removed "as v1 chose" (least-norm maximizer).
- `SubgradientDescent` [covered D-55] removed "v1's implementation uses a constant step and says in its own docstring that convergence is not guaranteed". Migration note: v1's constant-step subgradient method is `rule="constant"`, kept and not the default.
- `SubgradientDescent` [stale] `"sqrt"` was called "square-summable-but-not-summable". `1/sqrt(k)` is not square-summable; that property belongs to `"inverse"` (`1/k`). Corrected. The `"constant"` rule was missing from the list and `rule:` called every rule "diminishing"; both fixed.
- `SubgradientDescent._minimize` comment [history] removed "evaluating it again here doubled the cost of every step".
- `BundleResult` [history] removed "`value` used to be `minimum`" (possibly v1-facing: if v1 called it `minimum`, this is a migration note).
- `BundleResult.gap` [cite D-13] removed "and that is D-13"; the sentence now says `LevelBundleMethod` gives a certified bound as `lower_bound`.
- `BundleResult.lower_bound` [v1] removed "v1 reported it as `f_low`". Migration note: v1 `f_low` becomes `BundleResult.lower_bound`.
- `ProximalBundleMethod` [history] removed "which is less than 'a genuine bound' as this used to say".
- `ProximalBundleMethod.__init__` (`tolerance:`) [stale] did not say the test is relative (`gap <= tol * max(|f|, 1)`). Added.
- `ProximalBundleMethod.__init__` (`qp_solver:`) [covered D-56] removed the Backus-dual measurement, "Clarabel is thirty times faster than the projected gradient and a thousand times closer to the primal reference", and the D-56 citation. The appendix figures (10.84 s against 0.37 s; 1.7e-8 against 6.5e-11) give about 260 times closer, not a thousand; the thousand is Clarabel against OSQP. Check D-56 wording if quoting.
- `ProximalBundleMethod.__init__` (`qp_solver:`) [v1] removed "An exact backend is what v1 had through its master QP".
- `ProximalBundleMethod.minimize` [stale] `Returns:` said "the gap that certifies it"; the proximal gap is not certified (see `BundleResult.gap`). Corrected.
- `ProximalBundleMethod.minimize` comment [history] removed "This is the part that was wrong": cuts were once taken at the center, so null steps duplicated cuts; across the Backus tests the median Gram condition number was 1e67 and the worst infinite. The reason for taking cuts at the candidate stays.
- `ProximalBundleMethod._gram` [history] "Rebuilding it cost ..." became conditional; removed the "`k^3` over a run" aside.
- `ProximalBundleMethod._solve_model` [stale] called the gap "a genuine lower bound on how much is left to gain, and the only stopping criterion here that means anything", contradicting `BundleResult.gap`. Corrected. "whose dimension is the data's" became "which may be large".
- `_minimize_on_simplex` [RECORD] removed measurements not in DECISIONS.md. Handing the subproblem to a general nonlinear solver cost fifty seconds per bundle minimization. Over 30 random bundles at 400 iterations, the accelerated method left a smaller KKT residual on 93 per cent, by a median factor of 3300; it ended behind on 2 of 30. Keeping the best iterate took the median from 557 to 3300 and the tenfold share from 87 to 93 per cent. Across the Backus tests the median `Q` condition number was 1e23 and the residual 4.8e-7 at the median, 3.5e-3 at the 99th percentile (the reason `tolerance` and `warn_above` are separate).
- `_minimize_on_simplex` [history] removed "It used to return silently" and [cite D-56] "now uses by default ... (DECISIONS.md D-56)".
- `_minimize_on_simplex` comment [history] removed "measured over 30 random bundles it did end worse than the plain method on 2 of them" (figure in the RECORD item above).
- `LevelBundleMethod` [stale] "solved ... by the simplex method": `_lower_bound` calls `linprog(method="highs")`. Now "by SciPy's HiGHS solver".
- `LevelBundleMethod` [v1] removed "A port of v1's `LevelBundleMethod`" and [covered D-13] "Its author's view on the API is to be sought before it is changed beyond the port". Added the dense-master size limit, which D-56 records and the docstring did not.
- `LevelBundleMethod.__init__` (`tolerance:`) [stale] now states the floor of one (`max(|upper|, 1)`).
- `LevelBundleMethod._lower_bound` [v1] removed "as v1 did" (the box of radius `1e3 (1 + max |center|)`). D-56 says only "keeps the serious-step test, warm start and box".
- `LevelBundleMethod._master` [v1] removed "as v1 did" (warm start at the center with `t == warm`).
- `LevelBundleMethod.minimize` comment [v1] removed "as in v1" (level equal to the best value while there is no bound).
- `LevelBundleMethod.minimize` comment [history] removed "The center used to move every iteration"; the reason stays, conditionally.
- `LevelBundleMethod.minimize` comment [history] removed "used to be reported as the full run".
- `ChambollePockSolver` [v1] removed "v1's version could only handle a ball, because it implemented the projection itself and raised for anything else" and "A port of v1's `ChambollePockSolver`". Migration note: v1 `ChambollePockSolver` accepted only balls; v2 takes any `ConvexSet` with a projection.
- `ChambollePockSolver.__init__` [stale] `Raises:` omitted the `ValueError` for a negative `theta`. Added.
- `PrimalKKTSolver` [RECORD] removed the measurement against `ChambollePockSolver` over sixteen directions at noise radius 0.05: fourteen agree to 1e-11, two (where `mu` hit the clip) to 2.4e-3, and v1 behaves identically down to the same multiplier. DECISIONS.md does not record the KKT route's accuracy limit or this comparison.
- `PrimalKKTSolver` [v1] removed "A port of v1's `PrimalKKTSolver`, and the solver behind its `sphere_dli_example`". (`work/sphere_dli_example.py` still exists in the repo.)
- `PrimalKKTSolver` [stale] the heading "The model space is never discretized" was misleading (it is always discretized here); now "No matrix is formed on the model space", noting the dense data-space matrix.
- `PrimalKKTSolver.__init__` (`noise:`) [stale] now says a nonzero center is silently ignored (see code issues).
- `PrimalKKTSolver.solve` comment [v1] removed "as v1 has" (the physically scaled start).
- `PrimalKKTSolver.solve` comment [covered D-82] removed "Seen on a cold start where the noise multiplier went to 1e24 ...; the answer was 2 per cent low and marked converged".
- `PrimalKKTSolver.solve` comment [v1] removed "v1 carries a ladder of fallbacks for exactly this" and [history] "disagrees with the primal route by tens of per cent".
- `LevelKKTSolver` [cite D-80] removed "(DECISIONS.md D-80)" from the cold-start sentence. The citation was to the wrong entry: D-82 records the cold-started minimisation per probe; D-80 does not. "Let Newton stop one step short and froze the residuals" became present-tense.
- `LevelKKTSolver` [stale] "the membership engine's objective" and "the bisection route" were undefined here; now name the feasible-set membership test and `BackusGilbertParker`'s primal bisection route.

### geometry/convex.py

- module docstring [v1] removed "v1 keeps these in three places: the sets in `subsets`, the support functions in `convex_analysis`, and nothing that ties an indicator to a proximal method. Tying them together is most of the modernization." Migration note: v1 `subsets` and `convex_analysis` are merged into `geometry.convex` and `numerics.convex`, and `ConvexSet.indicator()` is new.
- module docstring [v1] removed "v1's `HalfSpace.project` instead goes to the bounding hyperplane whichever side the point is on". The point is kept without the v1 attribution. Migration note: `HalfSpace.project` is now the metric projection and leaves feasible points alone (D-67).
- module docstring [stale] "Where there is no closed form -- the projection onto a general ellipsoid, for instance -- the set says so rather than offering an approximation" was out of date: `Ellipsoid.project` exists (Newton on the secular equation). The docstring now names the algorithmic projections (ellipsoid, Dykstra) and the refusals (Minkowski sum).
- `ConvexSet.from_support_function` [history] removed "This is how the feasible property set of §18.3 arrives" (a retired-journal section number); now says the Backus-Gilbert-Parker engines use it.
- `ConvexSet.from_support_function` (membership) [stale] removed "the feasible property set has one when its constraint sets are balls"; nothing in the library passes `membership=`, since `FeasiblePropertySet` carries its own membership engine.
- `ConvexSet.__add__` [history] removed "Route (b) of §18.3 returns a sum of two ellipsoids and has no simpler description"; it now says which capabilities the sum carries.
- `_OracleSet.polytope` [history] removed "(§18.4)".
- `Polytope` [history] removed "§18.4's requirement made into a type".
- `Polytope` [covered D-80] removed "the mistake BGP's Figure 4 is about" (D-80: "reporting the inner hull alone is BGP's Figure 4 mistake"); the point is kept in plain words.
- `Polytope.project` (max_iterations) [RECORD] removed "A thousand rather than the two hundred this used to allow: measured on a ball cut by a half-space in a sixteen-dimensional Sobolev space, two hundred cycles leave the answer 3.7e-4 off the boundary and a thousand reach it exactly". DECISIONS.md does not record the Dykstra cap of 1000 or this measurement.
- `Ball.boundary` [v1] removed "v1 had this". Migration note: v1's ball boundary is `Ball.boundary`, a `BallSurface`.
- `HalfSpace.project` [v1] removed "which is what v1 does" (projecting onto the boundary regardless); covered as a convention by D-67.
- `HalfSpace.support_function` [v1] removed "v1's `HalfSpaceSupportFunction`". Migration note: v1 `HalfSpaceSupportFunction` becomes `HalfSpace.support_function()` / `SupportFunction.of_half_space`.
- `Ellipsoid` [stale] the class docstring said the projection "is not offered under the same name as the closed forms"; it is offered (`Ellipsoid.project`, Newton on the secular equation). Corrected.
- `Ellipsoid.__init__` (factor) [v1] removed "which is v1's `inverse_sqrt_operator` route in a form any factor can take". Migration note: v1 `inverse_sqrt_operator` becomes `Ellipsoid(..., factor=L)` with `C == L L*`.
- `Ellipsoid.support_maximizer` [stale] `Raises:` said `ValueError`; the code raises `AttributeError`, and the docstring now says so. The base class's contract is `NotImplementedError` (see CODE_ISSUES.md).
- `_OracleSet.maximizer` [stale] `Raises:` said `NotImplementedError`; the code raises `AttributeError`, and the docstring now says so (same code issue).
- `Ellipsoid.boundary` [v1] removed "v1 had this."
- `Ellipsoid.project` [history] removed "which is why this used to raise. It raises no longer, because a set that cannot project cannot be used by anything that needs a proximal step ... Costing a few solves is a better answer than not being available."
- `Ellipsoid.project` [covered D-67] removed "A Cholesky factorization used to be the default ... at dimension 1500, 3.5 s with a matrix-backed precision and 31 s with 54 000 applications of one that had to be probed (DECISIONS.md D-67)". The figures are in D-67 and the appendix; the docstring keeps only that a direct solver can be passed on a small space.
- `Ellipsoid.support_function` [stale] Raises said `ValueError`; the code raises `NotImplementedError`. Corrected.
- `BallSurface.contains` [history] removed the account of the keyword once being `tolerance` and raising `TypeError` inside `Intersection`, `Union` or `Complement`.
- `EllipsoidSurface.contains` [history] removed "Named `rtol` to match the abstract signature ... see `BallSurface.contains`".
- `ConvexIntersection` [history] removed "v2 was returning a plain `Intersection` for it"; the docstring now states why the class exists.
- `ConvexIntersection` [v1] removed "v1 offered that [min_i h_i] as the support function". Migration note: v1's intersection support function was the upper bound `min_i h_i`; v2 calls it `support_bound` and `support_function` raises.
- `ConvexIntersection` [stale] "the minimizing point of one set" should be the maximizing (support-attaining) point. Corrected.
- `ConvexIntersection.project` [history] removed "The default is a thousand for the reason given in `Polytope.project`" (that reason is now in the ledger above).
- `ConvexIntersection.support_bound` [stale] the example of "a part with no support function -- a half-space" was wrong: `HalfSpace` has an (extended-real) support function. Rewritten: an ellipsoid without its covariance is the example, the reason it may be left out is stated correctly, and a half-space's `+inf` contribution is described.
- `ConvexIntersection.support_function` [v1] removed "which is what v1 returned under this name".

### geometry/subspaces.py

- module docstring [stale] listed a third construction "from an explicit matrix -- only where coordinates exist", which does not exist in the module; and "Coordinate-free, since the operator involved is `A A*` and so positive semidefinite" was a non-sequitur. Rewritten to list the two constructions (basis, equation/kernel) and the hyperplane case.
- `_normal_inverse` [history] removed "each used to build its own ... with a direct one it is the matrix of `A A*` extracted and factorized again, three times over"; it now says the shared inverse means one factorization.
- `OrthogonalProjector.onto_kernel` [stale] "The palindrome rule" was undefined jargon; reworded as the traits the composition `A A*` earns.
- `AffineSubspace.with_translation` [v1] removed "v1 does the same". Migration note: none needed, the behaviour matches v1.
- `AffineSubspace.projection_operator` [stale] "is affine exactly when the translation is nonzero" was wrong: the offset `(I - P) t` vanishes whenever `t` lies in the tangent space. Now "has a nonzero offset exactly when the subspace does not pass through the origin".
- `LinearSubspace.from_kernel` [history] removed "It did not before, and a kernel is the commonest way such a subspace gets built."

### geometry/__init__.py

- No changes.

### geometry/sets.py

- `Complement.contains` (rtol) [stale] the description of the reversed tolerance was garbled ("counts as inside its complement only if it is outside to within this"); rewritten to say a point within the tolerance outside the original is still in it, and so not in the complement.
- `SublevelSet` [v1] removed "v1's `open_set` flag gave the strict inequality, and with it a closure and a warning whenever an open set met an operation needing a closed one" (covered as a decision by D-69). Migration note: v1 `SublevelSet(open_set=True)` has no v2 equivalent; every set is closed.
- `SublevelSet` [stale] `:mod:`.convex`` made fully qualified.
- `LevelSet.boundary` [v1] removed "v1 returned the empty set" (covered by D-69 in spirit; D-69 does not mention this specific change). Migration note: v1 `LevelSet.boundary` returned the empty set; v2 returns the level set itself.

### testing.py

- module docstring [v1] removed "In v1 the equivalents are mixins in the MRO of every space and operator (`class HilbertSpace(ABC, HilbertSpaceAxiomChecks)`), which puts a test suite inside the objects under test." (covered as a decision by D-105). Migration note: v1's `*AxiomChecks` mixin methods become the free functions `pygeoinf2.testing.check_*`.
- module docstring [cite D-128] removed "(D-128)".
- module docstring [history] removed "Every `rtol` defaults to what the check required before it was a parameter"; replaced with what the defaults are for.
- module docstring [stale] added the explanation of probe vectors (`random` unless `measure=`), which `_draw` used to carry with the wrong wording (below).
- `_draw` [stale] said the default probe is "white noise on the space"; the code draws `space.random`, which is not white noise. Corrected.
- `_draw` [v1] removed "v1's `measure=` on every check" (covered by D-105). Migration note: v1's `measure=` on the checks is kept under the same name.
- every `check_*` (measure) [stale] "rather than as white noise; see :func:`_draw`" was wrong (the default is `random`) and pointed at a private function. Rewritten as self-contained text in eleven places.
- `check_space` (rebuild) [cite D-20] removed "See DECISIONS.md D-19." The citation was also wrong: D-19 is white noise, while identity from `_key` is D-20.
- `check_representer` [cite D-26] removed "See DECISIONS.md D-26."
- `check_white_noise` [covered D-125] removed the account of how a fixed tolerance behaved on large and small metrics ("a correct space failed by chance, the off-diagonal entries first ... nothing could fail, the v1 construction included. One `rtol` now serves a metric of any size.").
- `check_white_noise` [v1] removed "v1 fails this check on every mass-weighted space" (covered by D-19); the point is kept as a general statement about drawing standard normal components.
- `check_white_noise` [cite D-19] removed "See DECISIONS.md D-19."
- `check_white_noise` (measure) [stale] the argument is used only for the test directions on a space without coordinates; the docstring now says so.
- `check_operator` [cite D-128] removed "(D-128)".
- `check_gradient` [cite D-26] removed "See DECISIONS.md D-26."
- `check_projection` [v1] removed "v1's `HalfSpace.project` satisfies neither of the first two" (covered by D-67).
- `check_projection` [stale] the docstring listed "the identity on the set" as a checked property and spoke of an "obtuse-angle condition" and a "variational inequality". The code checks that the projection lands in the set, that it is idempotent, and that no sampled nearby point of the set is closer. The list, the `probes` argument and `Raises` now say that.
- `check_convexity` [stale] "zero for a strictly convex functional" should be zero for any functional with no violating pair. Corrected.
- `check_affine` [v1] removed "-- v1's affine checks never looked there".

### plotting/sets.py

- module docstring [v1] removed "The compact port of v1's `SubspaceSlicePlotter`" (covered by D-94). Migration note: v1 `SubspaceSlicePlotter` becomes `plotting.plot_set(subset, subspace=...)`.
- module docstring [cite D-94] removed "(DECISIONS.md D-94)".
- module docstring [v1] removed "The exact polyhedral and quadratic paths v1 special-cased fall out of the first two routes"; the docstring now says polytopes and ellipsoids are drawn through the support or level function they offer.
- module docstring [history] removed "Three dimensions -- dropped in D-94, restored for the 2.0 release --" (covered by D-133).
- module docstring [stale] replaced the pointer to the private `_support_2d` and `_support_hull_3d` docstrings with the statement itself (the 3-D support picture errs inside the set, the 2-D one outside).
- `_support_hull_3d` [v1] removed "-- v1's route --" for intersecting the supporting planes (covered by D-133, which does not name v1's route). Migration note: v1 drew 3-D sets by intersecting supporting half-spaces (an outer bound); v2 draws the inscribed hull of attaining points.

### probability/gaussian.py

- module docstring [cite D-36] removed "(DECISIONS.md D-36)"; "adjoint-palindrome rule" replaced by a plain statement of what the algebra recognises (`A C A*` with `C` PSD).
- module docstring [cite D-19] removed "Sampling is where the white-noise correction of DECISIONS.md D-19 earns its keep".
- module docstring [v1] removed "v1 draws standard normal components there and so produces covariance `sigma^2 G` rather than `sigma^2 I` on any mass-weighted space". Migration note: v1's isotropic Gaussian had covariance `sigma^2 G` on weighted spaces; v2's is `sigma^2 I` (covered by D-19).
- `_semidefinite_factors` [v1] removed "exactly as v1 did" (clipping small negative eigenvalues with a warning).
- `_check_round_trip` [RECORD] removed measurement: CG at `rtol=1e-8` on a covariance of condition 4e10 returned a trace of 217 against a true 289 without complaint. DECISIONS.md has no entry for the round-trip check on the stochastic KL divergence's reference solve.
- `GaussianMeasure.from_standard_deviation` [v1] removed "This is the construction v1 gets wrong on a mass-weighted space." (covered by D-19).
- `GaussianMeasure.from_standard_deviations` [v1] removed: v1's method of this name built its factor as a map from a Euclidean coefficient space, so the array described components rather than directions; the two agree only on an orthonormal space. Migration note: v1 `from_standard_deviations` factor domain was `EuclideanSpace(n)`; v2's is the space itself.
- `GaussianMeasure.from_standard_deviations` [cite D-19] removed "the defect DECISIONS.md D-19 exists to record".
- `GaussianMeasure.from_standard_deviations` [stale] "The covariance is `diag(sigma^2)` as an operator on the space" holds only on a diagonal metric. On a non-diagonal metric `diag(sigma)` is not self-adjoint and `L L*` is not diagonal; checked numerically on `make_dense_metric_space(4)`. The docstring now says what the code does (see the code issue).
- `GaussianMeasure.from_product` [history] shortened the Woodbury-surrogate motivation for carrying the precision ("a solve nested inside a preconditioner inside a solve").
- `GaussianMeasure.from_covariance_matrix` [cite D-25] removed "because no trait implies it (DECISIONS.md D-25)"; the two forms are now defined in place.
- `GaussianMeasure.from_covariance_matrix` [v1] removed "v1 took a symmetric eigendecomposition, clipped small negative eigenvalues to zero with a warning, and built both factors from it; this does the same."
- `GaussianMeasure.from_covariance_matrix` [history] removed "v1 attached one and v2 had stopped doing so, leaving mahalanobis_squared, log_density and grad_log_density refusing on every measure built this way."
- `GaussianMeasure.from_covariance_matrix` [stale] the `form` Args entry said "No default", but the signature defaults to `"galerkin"`. Corrected, and the entry now advises passing it explicitly.
- `GaussianMeasure.from_covariance_matrix` comment [v1] removed "which is v1's version of this line" (v1 dropped the Gram matrix in the precision factor, right only on an orthonormal basis).
- `GaussianMeasure._draw_through_factor` [v1] removed "v1 carried a Karhunen-Loeve closure on its invariant measure class to avoid that, and lost it whenever the algebra rebuilt the measure" (covered D-61).
- `GaussianMeasure.hilbert_schmidt_norm` [v1] removed "which is v1's exact route".
- `GaussianMeasure.hilbert_schmidt_norm` [history] removed "it used to be the default".
- `GaussianMeasure.hilbert_schmidt_norm` [cite D-63] removed "(DECISIONS.md D-63)".
- `GaussianMeasure._spectral_slices` [v1] removed "v1 had these formulas on its correlated measure".
- `GaussianMeasure.nuclear_norm` [v1] removed "v1's exact route".
- `GaussianMeasure.nuclear_norm` [cite D-63] removed "(DECISIONS.md D-63)".
- `GaussianMeasure.kl_divergence` (dense_limit) [RECORD] removed measurement: at dimension 4000 the dense divergence is exact in 4.4 s, where a hundred probes take 37 s to reach +/- 1.5. D-50 records the 4000 limit and the caution about it, but not these timings.
- `GaussianMeasure.kl_divergence_estimate` (dense_limit) [RECORD] removed measurement and history: the limit "used to be" 512; dense 0.09 / 0.63 / 4.4 s against a hundred probes' 1.4 / 9.3 / 37 s at dimensions 960 / 2000 / 4000. Not in DECISIONS.md or the appendix.
- `GaussianMeasure.kl_divergence_estimate` [RECORD] removed accuracy measurement of the folded stochastic route: on a dense reference of dimension 400 and condition 1e6, 15.14 +/- 0.30 against an exact 15.67, where the old three-estimate route gave -7.0 +/- 22.8. D-63 records the refusal under "auto", not this evidence.
- `GaussianMeasure.kl_divergence_estimate` [history] removed "see _kl_similarity for ... what this replaced" (a public docstring pointing at a private method's history) and the comment "the old route returned such numbers routinely".
- `GaussianMeasure._kl_similarity` [RECORD] removed the history and measurements behind folding the divergence into one trace of `g(M) = M - 1 - log M`: the old route estimated `tr(C_o^-1 C_s)` and both log-determinants separately, and `KL(mu || mu)` at dimension 289 came out as 369 +/- 35, or -351 +/- 33 with the reference precision to hand. DECISIONS.md has no entry for the single-trace construction or for the solver round-trip check (CG at rtol 1e-8, condition 4e10: trace 217 against 289). The docstring keeps a one-sentence reason.
- `GaussianMeasure._kl_similarity` [history] removed "which is what the old route did to the two covariances, on operators that had not been checked"; "palindromes" replaced by "the form `B* A B`".
- `GaussianMeasure.rescale_directional_variance` [v1] removed "as v1 does it".
- `GaussianMeasure.rescale_directional_variance` [stale] the Raises entry claimed a non-positive target variance is refused; the code checks only the current variance along the direction. Corrected (see the code issue).
- `GaussianMeasure.with_dense_covariance` [v1] removed "it is v1's `with_dense_covariance(parallel=, n_jobs=)`, which had no successor here". Migration note: v1 `with_dense_covariance(parallel=, n_jobs=)` becomes `with_dense_covariance(n_jobs=)`.
- `GaussianMeasure.with_dense_covariance` [stale] Raises listed ValueError for a space without coordinates; `require_coordinates` raises TypeError. Corrected.
- `GaussianMeasure.with_dense_covariance` comment [v1] removed "That is v1's move here too" (replacing `L L*` by the assembled matrix).
- `GaussianMeasure.sparse_covariance` [cite D-65] removed "(DECISIONS.md D-65)".
- `GaussianMeasure.credible_set` [cite D-64] removed "The hardening of DECISIONS.md D-64"; "hardening" is now defined in place, with the other two hardenings named.
- `GaussianMeasure.credible_set` [history] removed "It used to be taken silently, with an O(N^3) inverse of the Galerkin matrix (DECISIONS.md D-64)" (covered D-64).
- `GaussianMeasure.credible_set` [cite D-131] removed twice (`unsafe` Args and Raises); the gate is now described as `domain.is_function_space`.
- `GaussianMeasure.credible_set` comments [history] removed "and used to raise" (precision-only measures) and "until it was corrected by two factors of G" (the dense Galerkin inverse; covered D-64).
- `GaussianMeasure.weakened_ellipsoid` [v1] removed "v1's `weakened_ellipsoid`; its `theta` is `power` here." Migration note: v1 `weakened_ellipsoid(theta=)` becomes `weakened_ellipsoid(power=)`.
- `GaussianMeasure.weakened_ellipsoid` [cite D-131] removed from the `unsafe` Args entry and from the comment "the one that carries D-131's gate"; Raises now mentions the gate at power 1.
- `GaussianMeasure.ambient_ball` [history] removed "§18.1's conversion", a reference to a section of the retired DESIGN.md journal.
- `GaussianMeasure.ambient_ball` [v1] removed "three of them from v1", "as v1's `LowRankEig` route", "which is v1's `radius_method="sampling"`". Migration note: v1 `radius_method="sampling"` becomes `method="sampling"`; v1's LowRankEig route becomes `method="spectral", rank=`.
- `GaussianMeasure.ambient_ball` [history] removed "and it used to go the long way round like everything else" (diagonal route).
- `GaussianMeasure.ambient_ball` [stale] Returns said "A ball containing the credible region". The ball and the credible ellipsoid carry the same probability and neither contains the other (D-64). Corrected.
- `GaussianMeasure.ambient_ball` comment [RECORD] removed timings for the generalized symmetric eigenproblem `(C_gal, G)` against a non-symmetric decomposition of `C_c`: 0.05 s against 0.26 s at dimension 1000, 4.0 s against 15.2 s at 4000. Not in DECISIONS.md.
- `GaussianMeasure._quantile_method_for` [RECORD] removed measurements behind the auto switch from Imhof to the moment-matched chi-square: Imhof 0.8 s at 500 weights and 10.4 s at 2000; matched agreement 9e-5 relative on 500 and 3e-5 on 2000, in 0.001 s. D-58 covers Imhof against the saddlepoint, not this "matched" switch (more than 1000 positive weights, effective rank above 100), which is undocumented in DECISIONS.md. The thresholds are now stated in the docstring.
- `GaussianMeasure.as_multivariate_normal` comment [history] removed "— which is what this was" (the broadcasting bug).
- `GaussianMeasure.condition` [history] removed "This used to be a dense `np.linalg.inv` with no way to change it."
- `GaussianMeasure.marginal` comments [history] removed "The trait deduction used to refuse the square root here for its own reasons; it no longer does, so the condition is stated where it belongs." and "(audit row on `marginal`)".
- `GaussianMeasure._block_of` comment [history] reworded "the marginal of a posterior on a direct sum claimed nothing and was refused" as a conditional guard.
- `GaussianMeasure.mahalanobis_squared` [cite D-131] removed twice.
- `GaussianMeasure.mahalanobis_squared` [stale] Raises said ValueError for a missing precision; the code raises NotImplementedError. Corrected.
- `GaussianMeasure.log_normalizing_constant` [cite D-131] removed twice.
- `GaussianMeasure._compute_log_normalization` comment [stale] "Taken before any retraiting" was obscure; reworded.
- `GaussianMeasure.log_density` [cite D-131] removed twice; added Returns and Raises (NotImplementedError without a precision, ValueError at the gate).
- `GaussianMeasure.grad_log_density` [cite D-26] removed "That is the whole content of DECISIONS.md D-26 in its most agreeable form."
- `GaussianMeasure.grad_log_density` [cite D-131] removed.
- `GaussianMeasure.grad_log_density` [stale] Raises said ValueError for a missing precision; the code raises NotImplementedError. Corrected, and the gate's ValueError added.
- `GaussianMeasure._rebuild` [history] removed the account of no operation keeping a precision (`mu.translate(v).log_density` raised; the spectral KL route fell back to a dense solve).
- `GaussianMeasure._diagonal_inverse` comment [history] reworded "Gating both on positivity lost the factor" as a conditional guard.
- `GaussianMeasure._mapped_precision` [history] removed "dropping the precision there was pure loss".
- `GaussianMeasure._combine_scale` [history] removed "Requiring a factor turned a measure described by a covariance and a precision ... into an unspecialized pushforward".

### probability/base.py

- module docstring [cite D-59] removed "See DECISIONS.md D-59."; "a direct dividend of Riesz-identifying the spaces" reworded.
- `_require_density` [cite D-11] and [cite D-131] removed; the guard is now described in its own words.
- `ProbabilityMeasure.sample` [v1] removed "v1 draws from NumPy's legacy global state, so no result involving sampling is reproducible." Migration note: v1 sampling used NumPy's global RNG; v2 takes `rng=` (a `Generator`) everywhere (covered D-59). Added an `rng` Args entry.
- `ProbabilityMeasure.two_point_covariance` [stale] Raises said NotImplementedError; the code raises TypeError. Corrected; Args and Returns added.
- `ProbabilityMeasure.log_density` [cite D-131] removed twice.
- `ProbabilityMeasure.grad_log_density` [cite D-131] removed twice.
- `ProductMeasure` [v1] removed "v1 has this only for Gaussians (`GaussianMeasure.from_direct_sum`)". Migration note: v1 `GaussianMeasure.from_direct_sum` becomes `GaussianMeasure.from_product` / `probability.product`.
- `ProductMeasure` [history] removed "and so is exactly the case the nonlinear work runs into".

### probability/mixture.py

- module docstring [history] removed "the evidence of §26", a retired DESIGN.md journal section; it now names `LinearGaussianInversion.log_evidence`.
- module docstring [cite D-66] removed "See DECISIONS.md D-66."
- `GaussianMixture.log_density` [cite D-131] removed twice; Raises now also names the no-density case the code raises for.
- `GaussianMixture.marginal_probabilities` [cite D-131] removed twice.

### probability/absolutely_continuous.py

- module docstring [cite D-132] and [cite D-131] removed.
- `AbsolutelyContinuousMeasure` [cite D-132] removed from the summary line; added a cross-reference to `GaussianPosterior`.
- `AbsolutelyContinuousMeasure.__init__` comment [stale] "the posterior diamond" and "The same accommodation Functional's positional codomain makes" were jargon and a vague cross-reference; the comment now names `GaussianPosterior`.
- `AbsolutelyContinuousMeasure.relative_log_density` [history] removed "The name is not the brief's bare `log_density`, deliberately" (the sampling brief); covered D-132, which records this naming.
- `AbsolutelyContinuousMeasure.relative_log_density` [cite D-131] removed "(D-131 gates it)".

### probability/__init__.py

(no changes)

### plotting/distributions.py

- module docstring [v1] removed the v1 to v2 renaming table. Migration note, verbatim: `prior_measure` -> `prior`; `true_values` -> `truth`; `posterior_labels` -> `labels`; `fill_density` -> `fill`; `num_sigmas` -> `sigmas`; `width_scaling` -> `width`; `contour_color` -> `color`; `plot_1d_distributions` -> `plot_densities`; `plot_corner_distributions` -> `plot_corner`.
- module docstring [history] removed "the randomize-then-optimize sampler of §18.7" (a retired journal section).
- module docstring [covered D-93] removed "by 75% on the weighted space in the test suite".
- module docstring [cite D-93] removed "See DECISIONS.md D-93."
- `_grid` [stale] "~160 points per standard deviation regardless of the ratio" holds only at `plot_densities`' defaults (2000 points, width 6); `plot_corner` uses 500 points and width 3.75, about 67. Now stated as `points / (2 width)`.
- `_binned_density` [history] removed "and about five seconds" (the gaussian_kde cost at 20 000 draws on 160 x 160).
- `_binned_density_2d` [RECORD] removed accuracy figures: an axis-aligned filter is wrong by 8% of the peak and moves the contour levels by 6% on a correlated cloud; the full kernel is wrong by 1.5% and 0.7%. The binned KDE has no DECISIONS.md entry (D-93 says only "histograms with kernel density").
- `plot_densities` [v1] removed "v1's `plot_1d_distributions`".
- `plot_densities` [history] removed the `title` justification "Every pyslfp call passes one, and without it a caller has to reach past the return value to set it."
- `plot_densities` [stale] Raises said ValueError for an index out of range and for a measure that can be neither read nor sampled; the code raises IndexError and TypeError (from `moments`). Corrected; the zero-variance ValueError added.
- `plot_densities` comment [stale] and [v1] the comment above the window computation said "v1 asks for 25 points per standard deviation of the narrowest peak; so does this". The code does not do that: `_grid` refines each curve to about 160 points per standard deviation. Replaced with a description of what the code does.
- `plot_corner` [v1] removed "v1's `plot_corner_distributions`".
- `plot_corner` (density) [RECORD] removed timings: binned is about 2000 times faster than `gaussian_kde`, 5.0 s against 2.2 ms per panel at 20 000 draws on a 160 x 160 grid, and 8.9 s of example 26's 12.2 s. Not in DECISIONS.md. The accuracy statement (1.5% and 0.7%) stays, since it informs the choice.
- `plot_corner` [history] removed "Every pyslfp call passes one" (title) and "v1 had one; v2 lost it" (legend).
- `plot_corner` comment [history] reworded the fill comment as a conditional and removed "so the same argument read opposite ways in one figure".

### plotting/base.py

- `show` [history] removed "which is six warnings in a suite run and no figures either way" and the reference to the test suite.
- `plot_network` [v1] removed "v1's `plot_geodesic_network`" and "v1's styling". Migration note: v1 `plot_geodesic_network` becomes `plotting.plot_network(space, paths, sources=, receivers=)`, with the same gold-star and red-circle styling (covered D-92).

### plotting/__init__.py

(no changes)

### inference/__init__.py

- module docstring [cite D-71] removed "See DECISIONS.md D-71 to D-82."; the example now also shows `BackusGilbertParker`, with its comments aligned.

### inference/estimators.py

- module docstring [cite D-72] removed "See DECISIONS.md D-72."
- module docstring [covered D-71] removed "and that rule is what says these three exist and stops a fourth being invented."
- `LinearPointEstimator` [covered D-72] removed "It joins the existing algebra rather than wrapping it" and "This is the *output* of a Backus-Gilbert method, not a diagnostic added afterwards."
- `LinearPointEstimator.as_measure` [history] removed "One of the bridges of §18.7: a linear point estimator, a Gaussian estimator and a set estimator are three readings of the same object, not three implementations." (§18.7 is a section of the retired design journal.)
- `LinearPointEstimator.as_measure` [stale] "Mean `B d`" left out the translation; now "`B d + t`". Added the `AttributeError` it raises through `propagated_covariance`.
- `GaussianEstimator.__init__` [covered D-73] removed "Carried here rather than attached by a subclass's `__call__` ... a sampler that only exists on the way out is a sampler `push_forward` silently drops." Added `Raises: ValueError`.
- `GaussianEstimator.__call__` [history] removed "which is just as well, since forming one on a model space is exactly what the whole arrangement avoids."
- `GaussianEstimator.push_forward` [stale] the opening paragraph ("Worth reaching for even when the model-space posterior is available, since `T C T*` on a small property space is cheaper than forming `C` on the model space at all") contradicted the "Which of the two to reach for" paragraph after it, and implied that `C` is formed. It now says the covariance is the operator `T C T*`.
- `GaussianEstimator.push_forward` comment [stale] "the palindrome is no longer visible in it" was undefined jargon; the comment now says what the composition node recognises and why a two-factor covariance defeats it.

### inference/laplace.py

- module docstring [cite D-7] removed "(**D-7**)".
- module docstring [history] removed "The one substantive capability absent from both versions".
- module docstring [covered D-7] removed "**Built so a sampler can replace step 2.** ... adding such a sampler later means adding the sampler and nothing else" (planning). The paragraph now says what a sampler finds here, including the exact relative density that `at` carries.
- `MaximumAPosteriori` [stale] `solver` said "Left to `LinearGaussianInversion`'s own default if omitted"; the code calls `resolve_solver` directly, which gives CG. It now says so, and that a factory callable is accepted. The `optimizer` default is now stated as L-BFGS with 500 iterations.
- `MaximumAPosteriori` [stale] Raises left out the `TypeError` that `problem.error_measure` raises for a set-valued error.
- `MaximumAPosteriori.log_posterior` [history] removed "and because writing `-0.5 *` at the call site is exactly the kind of factor that goes missing."
- `MaximumAPosteriori.linearized_normal` [stale] "the curvature of `objective` up to a factor of two -- up to the Gauss-Newton approximation" was garbled; now "half the Gauss-Newton approximation to the Hessian of `objective`".
- `MaximumAPosteriori.at` [cite D-132] removed "(D-132)". `-Phi` is now defined through `ForwardProblem.potential`, and "Gaussian reference" became "Gaussian approximation", since the reference measure is the prior.

### inference/mixture.py

- module docstring [history] removed "with everything §23 through §28 gives them" (journal sections); it now names what each component inversion carries.
- module docstring [cite D-66] removed "See DECISIONS.md D-66."
- `LinearGaussianMixtureInversion.inversions` [stale] "its gain" became "its Kalman operator" (the attribute is `kalman_operator`); removed "there is no reason to hide them behind the mixture".
- `_PushedMixture.__call__` [history] removed "With the weights needing the same solve again, that was `2K` solves for `K` components where `K` will do."

### inference/backus.py

- module docstring [cite D-79] removed "See DECISIONS.md D-79."
- module docstring [stale] said the module holds route (a), the closed form, and route (b), the linear certificate, and that routes (c) and (d) are "§18.3's expensive general cases", as if they lived elsewhere. Every route is in this module, and so are the KKT route and the likelihood membership. The docstring now lists the routes `BackusGilbertParker` chooses from, and defines "BGP".
- `_self_adjoint_spectrum` [history] removed "(§5.6)" and "which is why the distinction stays invisible until it isn't"; the eigenproblem was written `M v == lambda G v` with `M` undefined, and is now `(G N_c) v == lambda G v`.
- `harden_error` [history] removed the account of the rule it replaced, `sqrt(chi2_crit * mean diagonal of the component matrix)`: why that is wrong on a weighted data space (the components' covariance is `C_c G^-1`), and that it formed a dense matrix to read one scalar.
- `BackusGilbert` [history] removed "in the sense of §18.3(b)".
- `BackusGilbert` [stale] "What comes out is not a number but a set": `__call__` returns `X d`, and the set comes from `uncertainty(data)`. Rewritten, with `M` and `D` defined as the two radii.
- `_ClosedFormRoute` [history] removed "Route (a) of §18.3" and "Every piece of that is already here: `A A*` is positive semidefinite by the palindrome rule, `onto_kernel` is example 18's subject, and `Ellipsoid` carries the support function the bound's directional form needs."
- `_ClosedFormRoute.is_feasible` [v1] removed "v1's `test_data_compatibility`". Migration note: v1 `test_data_compatibility(...)` is `not BackusGilbertParker(...)(data).is_empty()`.
- `_ClosedFormRoute._joint_spectrum` [history] removed "Rebuilding it per call made every sweep over property values pay for a full eigendecomposition"; the reason for caching stays, in the present tense.
- `_ClosedFormRoute.inclusion_norm` [stale] removed "the palindrome rule gives no more than that" (undefined jargon).
- `_minimum_norm_fits` [v1] removed "v1's `test_data_compatibility`, matrix-free".
- `_minimum_norm_fits` [cite D-81] removed "(DECISIONS.md D-81)"; reworded "which at a datum per column is the one thing a data space of any size cannot afford", and stated that both balls are taken as centred at the origin.
- `_BisectionRoute` [history] removed "Route (c) of §18.3" and "the same primitive as §18.6" (now "the discrepancy principle's").
- `_BisectionRoute` [v1] removed "For balls both are scalings and the arithmetic is v1's." Migration note: for two balls the bisection does v1's primal arithmetic.
- `_BisectionRoute._data_gram` [stale] "each of the four thousand bisection steps per direction" is now "up to `max_iterations` squared per direction" (60 × 60 by default).
- `_BisectionRoute._bisect` [cite D-57] removed "DECISIONS.md D-57's one kernel".
- `_BisectionRoute.fitting_model` [v1] removed "For two balls this is v1's `test_data_compatibility` computation".
- `_BisectionRoute.fitting_model` [cite D-81] removed "(DECISIONS.md D-81)".
- `_BisectionRoute._property_pseudo_inverse` [history] removed "(§18.1)".
- `_DualRoute` [history] removed "Route (d) of §18.3"; "routes (a) and (c)" are now named, and the alternatives this object also answers by are mentioned.
- `_DualRoute` [v1] removed "which is exactly what v1's `DualMasterCostFunction` docstring writes down without naming it as a support function of an image." Migration note: v1 `DualMasterCostFunction` is `_DualRoute.dual_cost`, reached through `BackusGilbertParker(..., route="dual")`.
- `_DualRoute.dual_cost` comment [v1] removed "v1 fused them into one oracle for exactly this reason".
- `_DualRoute.primal_solver` [RECORD] removed "measured, they agree to 1e-10, which is strong duality checked rather than assumed". DECISIONS.md does not record the measured primal-dual agreement.
- `_DualRoute.support_values` [v1] removed "Restores v1's `solve_support_values`" and "This is v1's `solve_primal_feasibility`". Migration note: v1 `solve_support_values` is `FeasiblePropertySet.support_values(directions)` on the dual route; v1 `solve_primal_feasibility` is `route="primal"`.
- `_DualRoute.support_values` [RECORD] removed "Measured agreement between the three: 1.7e-8 relative between dual and primal, and 2.7e-11 between primal and KKT." Neither D-79 to D-82 nor the appendix records these figures.
- `_DualRoute.support_values` [stale] said "All three agree" and left out the `"smoothed"` route, which the code accepts. It said `"kkt"` needs balls or ellipsoids, although other level-function sets go to `LevelKKTSolver`; Raises said "not one of the two" though there are four routes; `**kwargs` were said to go to the primal solver alone, though the KKT and smoothed routes take them too. Also noted that the KKT route ignores `warm_start`.
- `_DualRoute.support_values` [history] removed "and the part `ProximalBundleMethod` does not supply on its own".
- `_DualRoute.certificate` [history] removed "(§18.3(b))", now "by weak duality".
- `_DualRoute.is_feasible` [v1] removed "this is v1's test".
- `_LikelihoodRoute.__init__` comment [history] rewrote the forcing-term comment in the present tense ("with the inexact default of 1e-3 the step carried ..."); the 1e-3 default and its effect stay.
- `_LikelihoodRoute._objective` comment [RECORD] removed "which left the fitted misfit 0.8 per cent above the level on one seed in twenty-five". The reason for scaling by the multiplier stays in the comment; the measurement is not in DECISIONS.md.
- `_LikelihoodRoute._fit` comment [history] removed "what the first version did" and the figure "0.7 per cent above the level on one seed in twenty-five"; the reason every probe starts cold (warm-started Newton-CG stops short and the search closes on a false root) stays.
- `_LikelihoodRoute._fit` comment [RECORD] removed "the optimizer's stopping rules on warm starts are an open point recorded in DECISIONS.md D-80". D-80 records no such point, so the open question (why warm-started Newton-CG stops a step short, and whether the stopping rules should change) is recorded nowhere.
- `_LikelihoodRoute.fitting_model` comment [stale] the comment on the dead `if ...: pass` branch ("The zero correction fits; ... so start it") was unclear; reworded. The dead branch is in the issues file.
- `FeasiblePropertySet` [stale] said a level function needs "a ball prior"; said a support function always exists, though there is none when no route applies; said a maximizer exists on the closed form and the bisection only, though the KKT route has one too.
- `FeasiblePropertySet.is_empty` [stale] said emptiness is "decided by the route"; the code asks the membership engine first and falls back to the support route.
- `FeasiblePropertySet.support_maximizer` / `.extremal_model` / `.certificate` [stale] the Raises clauses said "on a general route", but the KKT route, a general one, exhibits the maximizer; the no-route case was missing.
- `FeasiblePropertySet.polytope` [covered D-80] removed "(§18.4)"; the docstring now says what the bound is and that it should be reported beside an inner bound (BGP's Figure 4 point).
- `FeasiblePropertySet.inclusion_norm` [history] removed "§18.5".
- `FeasiblePropertySet.inner_hull` [stale] "The *inner* bound of §18.4, and the only thing that produces one": `extent` gives inner bounds too.
- `BackusGilbertParker` [history] removed "§18.3" and "§18.5".
- `BackusGilbertParker` [stale] the route list gave bisection for "a ball prior and a ball confidence set" (it takes a ball or an ellipsoid on each side) and left out the KKT route, which `"auto"` picks for differentiable level functions. The membership paragraph said it "needs a ball prior" and that a value is acceptable "within the prior radius", but the reduced and likelihood engines take ellipsoid and sublevel-set priors.
- `BackusGilbertParker.__init__` [stale] `route` pointed to "the dual sweep's own documentation", a private method; it now describes the four general solvers itself. `prior` said only a Ball admits "the cheap routes"; `membership` said `"reduced"` is for two balls; `optimizer` said Newton-CG is used when "the confidence set's functional" has a Hessian, though both sets' must; Raises lacked the `TypeError` for a prior or confidence set that is neither a `ConvexSet` nor a `SublevelSet`.
- `BackusGilbertParker._inclusion` [history] removed "the minimum-norm computation of §18.5".

Open documentation point (not a removal): neither "BGP" nor "Al-Attar (2021)" has a full bibliographic citation anywhere in pygeoinf2, and the retired journal has none either. The module docstring now says what BGP refers to; the references themselves need adding.

### inference/point.py

- module docstring [cite D-76] removed "see DECISIONS.md D-76" (the citation did not fit: D-76 is about `DiscrepancyPrinciple`, while the sentence was about the Tikhonov-based estimators).
- module docstring [stale] "Three methods, and a fourth that is one of them..." miscounted the five classes, and "each exposes the operator it inverts" was false for `DiscrepancyPrinciple` and `ConstrainedMinimumNorm`. Now lists each estimator and says which expose their normal operator.
- `LeastSquares.with_formalism` [history] removed "the test suite checks that they do"; now says the two formalisms agree to the solver's tolerance.
- `LeastSquares.residual_callback` [v1] removed "v1's `normal_residual_callback`". Migration note: v1 `normal_residual_callback` becomes `LeastSquares.residual_callback(data)`.
- `MinimumNorm.for_data` [stale] the second bullet said that when the misfit exceeds the threshold at every damping "the answer is the smallest"; the code raises `ValueError` there (`_searched_damping`, `exhausted == "low"`), and a later paragraph called the same case a "third case" that is refused. Now two cases: the largest damping is returned, or the call is refused.
- `misfit_search` [stale] removed "which is what makes a sixty-step search cost roughly one solve rather than sixty". D-57 records a warm start taking 6504 to 4892 inner iterations, about 25 per cent, not sixtyfold. Now says only that the warm start saves inner iterations.
- `_searched_damping` [covered D-76] removed the account of `for_data` returning a damping of 1e-200 with chi-squared 5e7 against a target of 9.5 while the principle raised. D-76 records the saturated-end semantics; the specific numbers were history.
- `_searched_damping` comment [v1] removed "v1 raises here too." Migration note: v1 also raised when no damping fits.
- `DiscrepancyPrinciple.__init__` (minimum_damping) [history] "rather than after two hundred decades of probes" reworded to "rather than after probing ever smaller dampings".
- `DiscrepancyPrinciple._linearize` [history] reworded the "otherwise called ... ran twice" past tense into a statement of what the override avoids.
- `ConstrainedLeastSquares.parameterized` [v1] removed "That is v1's construction". Migration note: v1 pulled the constraint back as `(B M) p == w` too.
- `ConstrainedLeastSquares.parameterized` [history] removed "This used to refuse outright, on the grounds that the parameterization alone does not determine the constraint."
- `_parameterized_subspace` [v1] removed "That is v1's construction".
- `ConstrainedMinimumNorm.constraint_value_mapping` [stale] removed a trailing paragraph after `Raises:` that repeated the `ValueError` entry word for word.

### inference/gaussian.py

- module docstring [stale] "The Kalman update, assembled in whichever space is smaller" contradicted the `"data_space"` default (D-74); now says data space by default, model space on request.
- module docstring [cite D-72] and [cite D-44] removed "See DECISIONS.md D-72 and D-44."
- `GaussianPosterior` [cite D-132] removed "(D-132)"; defined `Phi` (the likelihood potential, half the chi-squared misfit); "the Laplace route" now names `MaximumAPosteriori.at`.
- `GaussianPosterior` [covered D-132] removed "pretending otherwise was D-73's warning" and the mention of `GaussianMeasure._rebuild` (a private method); the docstring keeps the user-facing fact that push-forwards and conditionings return a plain Gaussian.
- `LinearGaussianInversion.__init__` (formalism) [stale] said only that `"auto"` takes the smaller space; now names the `"data_space"` default and links `choose_formalism`.
- `LinearGaussianInversion.__init__` comment [history] removed "which is exactly what went wrong in backus.py" from the id()-key comment.
- `LinearGaussianInversion.kalman_operator` [v1] removed "v1's name". Migration note: v1's `kalman_operator` name is kept, now a property rather than a method.
- `LinearGaussianInversion.kalman_operator` [history] removed "the test suite checks it against both formulae".
- `LinearGaussianInversion.data_prior` [v1] removed "v1's `data_prior_measure`". Migration note: v1 `data_prior_measure` becomes `data_prior`.
- `LinearGaussianInversion.joint_prior` [v1] removed "v1's `joint_prior_measure`". Migration note: v1 `joint_prior_measure` becomes `joint_prior`.
- `LinearGaussianInversion.with_formalism` [history] removed "the test suite checks the agreement".
- `LinearGaussianInversion._data_key` [history] removed "a few microseconds against a solve of tens of milliseconds" on the Euclidean data space (a trivial measurement); now "small beside a solve".
- `LinearGaussianInversion._residual_solve` [v1] removed "v1 computed the mean as `m0 + K(d - shift)` and shared it".
- `LinearGaussianInversion._residual_solve` [history] removed "v2 had the mean inside an affine operator and the misfit in its own method, so ... cost two solves"; now states what the memo saves.
- `LinearGaussianInversion.__call__` [cite D-132] removed "(D-132)" and "the same Gaussian as always, now also carrying"; added that the normalizer is present only when the problem has an error measure (the code passes `None` otherwise).
- `LinearGaussianInversion.mahalanobis` [stale] "the unnormalized log posterior at the posterior mean" had the wrong sign and factor; the quantity is minus twice that, the minimum over models of misfit plus prior penalty.
- `LinearGaussianInversion.normal_log_determinant` [history] trimmed the motivation ("the one part of an evidence calculation that looks like it needs the matrix ... not where model comparison is interesting") to one sentence: the stochastic route is matrix-free, so the evidence is available on problems too large to assemble.

### inference/normal.py

- module docstring [cite D-44] removed "See DECISIONS.md D-44."
- module docstring [v1] removed "v1's diagonal preconditioner uses ..." and "which is why in v1 they had to live as methods on the inversion class, the only place that still held the parts". Migration note: v1's structure-aware preconditioners were methods on the inversion class; v2's are free-standing solvers that read the factors off the normal operator. The identity `<v, A Q A* v> == <A* v, Q A* v>` stays, as a plain statement.
- `choose_formalism` [stale] removed "See DECISIONS.md D-57 and D-78", which cited the wrong entries: the formalism default is D-74, while D-57 is root finding and D-78 evidence.
- `FactoredNormalOperator.scale` [covered D-44] removed the past-tense account ("without that it returned `t` times the inverse, which ... conjugate gradients never noticed"); kept as a present-tense contract for anyone writing a preconditioner from the factors.
- comment above `FactoredNormalOperator._known_matrix` [stale] "read, blocked and fused" was jargon; now names the hooks (known matrix, diagonals, block application, component actions).
- `NormalOperator.__init__` [RECORD] removed the measurement "mean difference 0.86 against a mean of norm 2.6 on a random 12-by-8 problem" for the false "noise-free" model-space posterior (`Q^-1 + A* A`, i.e. `R == I`). DECISIONS.md does not record that the model-space formalism refuses an error-free problem, or why.
- `NormalOperator.__init__` [v1] removed "v1 does exactly that, silently." Migration note: v1 assembled `Q^-1 + A* A` for a noise-free model-space problem, silently using `R == I`; v2 raises.
- `NormalOperator.__init__` [stale] the explanatory paragraph sat after `Raises:`; moved before `Args:`. `formalism` now states the data-space default.
- `NormalOperator.surrogate` [stale] "In the tomography example" named the wrong example; the coarse-sphere surrogate is example 24 (preconditioning).
- `NormalOperator.posterior_covariance` [history] trimmed the cost motivation (a third of the cost of every variance, credible interval and pushed-forward property) to one sentence.

### inference/preconditioners.py

- module docstring [v1] removed "in v1 they had to be methods on the inversion class for exactly that reason". Migration note: as for normal.py, v1's structure-aware preconditioners were inversion-class methods.
- module docstring [stale] "Both are for the data-space formalism. Neither identity..." but the module has three preconditioners, all refusing a model-space operator through `_require_normal`; and "These take a `NormalOperator`" now says `FactoredNormalOperator`, which also covers the Tikhonov one.
- `NormalDiagonalPreconditioner` [history] removed "The test suite checks that it does, which is what pins the metric handling"; kept why the Galerkin diagonal is the right one.
- `LocalisedPreconditioner` [stale] "a randomized Nystrom decomposition of `P N P*`" was wrong twice: the code uses `random_eig` (a Rayleigh-Ritz randomized eigendecomposition; D-53 dropped Nyström), and on the sub-block of `A Q A*`, not of `N`. The `rank` Args entry and the `restrict` comment ("which is what Nystrom needs") are corrected the same way.
- `LocalisedPreconditioner` [history] removed "(They used to be summed, because a COO assembly adds duplicates: an entry two blocks agreed on came out twice its own value.)"
- `LocalisedPreconditioner` [history] removed "and what the test suite checks" from the note on `R`'s diagonal.
- `LocalisedPreconditioner._invert` comment [cite D-26] removed "(DECISIONS.md D-26)".
- `InvariantDistancePreconditioner` [v1] removed "v1's version defaulted to no taper, which is the most likely reason it never performed as well as hoped". Migration note: v1's `taper` defaulted to off; v2 defaults it on.
- `InvariantDistancePreconditioner` [cite D-47] removed "See DECISIONS.md D-47."; kept the user-facing point that without a taper the preconditioner can be indefinite, which breaks CG.
- `InvariantDistancePreconditioner` [history] replaced "the test suite checks the assembled matrix against `A Q A*` directly" with advice to the caller to check against the exact inverse on a small case.
- `InvariantDistancePreconditioner._invert` comment [history] "the matrix was right only on an orthonormal data space, and the Galerkin noise was being added" recast as a conditional (what would go wrong without the `G` factors).

### inference/problem.py

- module docstring [cite D-72] removed "see DECISIONS.md D-72".
- module docstring [history] removed "the two data relations of section 18.1", a section of the retired design journal (DESIGN.md, in git at 83658d0). The classification itself is D-71.
- `ForwardProblem.chi_squared`, `chi_squared_from_residual` [stale] "the Mahalanobis distance" should be the *squared* Mahalanobis distance (the code returns `mahalanobis_squared`).
- `ForwardProblem.chi_squared_from_residual` comment [cite D-131] removed "past D-131's gate"; the comment now says what the gate is.
- `ForwardProblem.critical_chi_squared` comment [history] removed "about three hundred modules and a quarter of a second" from the lazy-import comment. The import-cost concern is already recorded in `tests/test_code_practice.py::TestImportCost`.
- `ForwardProblem.chi_squared_test` [history] removed "(§18.11)", a journal section.
- `ForwardProblem.potential` [cite D-132] and [cite D-127] removed "(D-132)" and "as D-127 arranges"; "so a sampler's gradient costs one solve" reworded to one forward evaluation for value and derivative together, and only where the forward has a derivative (the code falls back to value-only otherwise).
- `ForwardProblem.potential` [stale] `Raises:` said `ValueError` for a set-valued uncertainty; the code raises `TypeError`. Now lists both exceptions.
- `ForwardProblem.consistency_set` [history] removed "the hardening of §18.1"; now says what the set is (the error's credible ellipsoid at `level`, centered on the predicted data).
- `LinearForwardProblem.joint_measure` [stale] "drawn together rather than ... given an independent error afterwards, which would be the same thing only by accident" was wrong: the joint law is exactly a prior draw plus an independent error draw. Now says so, and that with an error measure both measures must be Gaussian (see the code issue).

### inference/tikhonov.py

- module docstring [cite D-75] removed "See DECISIONS.md D-75."
- `TikhonovNormalOperator.prior_covariance` [history] removed "used rather than argued about".
- `TikhonovFamily.__init__` (solver) [stale] "Cholesky by default" was wrong: `resolve_solver(None, ...)` gives `CGSolver` (D-39). Also documented that a callable receives a representative member.

### symmetric_space/base.py

- module docstring [cite D-83] removed "See DECISIONS.md D-83."
- module docstring [v1] removed "the correction v1 writes by hand as `sqrt(spectral_variances / metric_values)`"; it now says white noise supplies the `1/sqrt(g)` factor. Migration note: v1 sampled invariant measures with a hand-written `sqrt(spectral_variances / metric_values)`; v2 gets the same factor from `white_noise` on a diagonal-metric space.
- `PreparedPoints` [RECORD] removed the timings that justify preparing points once (measured 2026-08-29): the conversion was 14.5 of 37 ms on a sphere at 1e5 points, 32 of 61 ms on a torus, and 230 of 254 ms (91%) on a bounded box, whose conversion is a Python loop. DECISIONS.md has no entry for `PreparedPoints` or for converting points once at construction.
- `SymmetricSpace.is_function_space` [cite D-131] removed.
- `SymmetricSpace.basis_at` [cite D-26] removed "See DECISIONS.md D-26."
- `SymmetricSpace.spectral_operator` [v1] removed "v1 spells this `InvariantLinearAutomorphism.from_index_function`". Migration note: v1 `InvariantLinearAutomorphism.from_index_function(f)` becomes `space.spectral_operator(f(space.degrees))`, which takes an array, not a callable.
- `SymmetricSpace.coefficient_operator` (lmax) [v1] removed "That is v1's zero-padding, with the adjoint derived rather than written." Migration note: v1 zero-padded past the truncation by hand; v2 composes with the prolongation.
- `SymmetricSpace.from_coefficient_operator` [v1] removed "v1 papered over the difference by writing `* radius**2` into the adjoint by hand, which is right for Lebesgue and wrong for every Sobolev order." Migration note: v1's synthesis adjoint was correct on Lebesgue spaces only.
- `SymmetricSpace.sufficient_degree` [v1] removed "v1's `estimate_truncation_degree`". Migration note: v1 `estimate_truncation_degree` is v2 `sufficient_degree`. v2 also has a method named `estimate_truncation_degree`, which answers a different question (see CODE_ISSUES).
- `SymmetricSpace.power_spectrum` [v1] removed "what v1 got at, on the sphere only, by drawing samples and calling `SHCoeffs.spectrum`" and "to recover v1's `sample_power_measure`". Migration note: v1 `sample_power_measure` becomes `power_spectrum` applied to a measure's samples.
- `SymmetricSpace.power_spectrum` [cite D-130] removed "(D-130)".
- `SymmetricSpace.power_spectrum` [stale] "On the sphere it is `4 pi` times what `SHCoeffs.spectrum` reports" holds only on the unit sphere; checked numerically, the ratio is `4 pi r^2` (50.27 at r = 2). Corrected.
- `SymmetricSpace.heat_symbol` [v1] removed "v1 spells this `heat_kernel(scale)` with the same convention." Migration note: v1 `heat_kernel(scale)` is `heat_symbol(length_scale)`, same convention.
- `SymmetricSpace.heat_symbol` [history] removed "v2 briefly took the time instead, silently"; kept the warning that a time passed as a length changes the correlation length (0.1 becomes about 0.32).
- `SymmetricSpace.pointwise_variance` [cite D-26] removed "the error of DECISIONS.md D-26 once more".
- `SymmetricSpace.invariant_measure` [history] removed "Going through the factor synthesizes white noise onto the grid only for a diagonal operator to analyze it again, at three transforms for the same numbers."
- `SymmetricSpace.invariant_measure` comment [covered D-61] removed "(measured 2026-08-29)" and "(DECISIONS.md D-61)".
- `SymmetricSpace.sobolev_measure` [stale] `Args:` documented `length_scale:`, but the parameter is `scale`; renamed in the docstring.
- `SymmetricSpace.sobolev_measure` [stale] "order ... must exceed half the spatial dimension for the draws to be functions" is right only on a Lebesgue space. On a Sobolev host of order t, a draw's L2 coefficient variance is `(1+scale²λ)^-order / (1+L²λ)^t`, so the host's order adds to it. Reworded.
- `SymmetricSpace.heat_measure` [v1] removed "which is v1's `heat_kernel_gaussian_measure(scale)` unchanged". Migration note: v1 `heat_kernel_gaussian_measure(scale)` is `heat_measure(length_scale)`, same convention (D-9).
- section comment before `correlated_measure` [stale] a "Point evaluation" header sat above the correlated-measure methods; moved it to `_require_point_evaluation`, where point evaluation starts.
- `SymmetricSpace.correlated_measure` [history] removed "the same argument as §13.1, one field wider" (a journal section number that no longer exists).
- `SymmetricSpace.correlated_measure` comment [history] reworded "through the factor as an operator it was ... three transforms" into the present-tense reason.
- `SymmetricSpace.spectral_correlations` [v1] removed "following v1" (zero where a marginal has no variance). Migration note: same convention as v1.
- `SymmetricSpace.power_measure` [cite D-130] removed "(D-130)" twice.
- `SymmetricSpace.power_measure` [cite D-11] removed "as with point evaluation under D-11".
- `SymmetricSpace.power_measure` comment [history] removed "the comprehension this replaces was O(dim^2), and 6.8 s against 0.4 ms at lmax 511."
- `SymmetricSpace.covariance_function` [covered D-11] removed "D-11's position -- the user's, answering Q4 --" and "(measured 2026-08-29)"; the guard is now stated as a rule.
- `SymmetricSpace.pointwise_variance_at` comments [history] removed "rather than five transforms per point (measured 2026-08-29)", "since the Dirac that used to raise it", and "it was, and was dead work".
- `SymmetricSpace.pointwise_variance_at` [stale] the docstring said exactly "Pass `samples` to estimate"; `rtol` alone also selects the sampled route, and `rank` applies only there. Both now said.
- `SymmetricSpace.pairs_within_distance` [v1] removed "as v1 does on the sphere". Migration note: v1 had the KD-tree search on the sphere only; v2 has it on every geometry.
- `SymmetricSpace.pairs_within_distance` [history] removed the timings of the superseded dense-difference implementation (576 MB and 0.21 s at n = 3000 against 7 MB and 0.08 s; 2.4 GB at n = 1e4) and "On the base rather than on the sphere, because".
- `SymmetricSpace.cluster_points` [history] removed "the one v2 dropped -- it offered a greedy radius rule seeded by the lowest remaining index, which is neither of v1's and is not stable under reordering the points"; kept, in the present tense, that the result does not depend on point order.
- `SymmetricSpace.cluster_points` [v1] removed "as in v1" from the `method` default. Migration note: same linkage default as v1.
- `SymmetricSpace.random_points` [history] removed "It lived on the sphere, so every other geometry lacked it for no reason anyone had decided."
- `SymmetricSpace._require_point_evaluation` [v1] removed "v1 refuses, v2 stopped refusing, and this puts it back." Migration note: v1 also refused point evaluation below `d/2`.
- `SymmetricSpace._require_path_evaluation` [covered D-83] removed the sphere representer-norm measurement and "(DECISIONS.md D-83)".
- `SymmetricSpace._require_path_evaluation` [v1] removed "v1 guarded this with the point threshold, safe and too strict by half a unit; v2 bypassed the guard with a comment naming the right condition." Migration note: path integrals accept orders down to `(d-1)/2`, where v1 required `d/2`.
- `SymmetricSpace.dirac` [stale] the sentence about the Dirac on Lebesgue and Sobolev spaces was tangled; restated with the threshold.
- `SymmetricSpace.point_evaluation_operator` (nthreads) [RECORD] removed "finufft's default of every core is slower here at every size measured". This measurement fixed the default `nthreads=1`, and DECISIONS.md (D-52, D-89) does not record it.
- `SymmetricSpace.point_evaluation_operator` comment [history] removed "(measured 2026-08-29)" and "thousands of times".
- `SymmetricSpace.from_grid_values` [v1] removed "as v1's `from_array` copies". Migration note: v1 `from_array` is v2 `from_grid_values`, and both copy.
- `SymmetricSpace.from_grid_values` [history] restated "aliasing it meant a later `axpy` ... silently rewrote it" in the present tense.
- `SymmetricSpace._own_grid_values` [history] removed "2 MB per call at `lmax` 256".
- `SymmetricSpace.truncate` [history] "`multiply` no longer applies it" became "does not apply it".
- `SymmetricSpace.multiply` [cite D-87] removed "(DECISIONS.md D-87)".
- `SymmetricSpace.multiplication_operator` [history] removed "(§3.5)" (old journal section) and "Truncating the product changed neither side of the identity; it only cost two transforms per application."
- `SymmetricSpace.multiplication_operator` [stale] "The claim survives" did not say which claim; it now names self-adjointness on L2.
- `SymmetricSpace.gradient_dot_product` [covered D-90] removed "v1 has it the other way round ... It went unnoticed because ..."; kept a sign warning in its own words.
- `SymmetricSpace.gradient_dot_product` [v1] Migration note: v1's `gradient_dot_product` had the opposite sign (D-90 records this).
- `SymmetricSpace.gradient_dot_product` [history] removed "for the same cost, and for two fewer transforms per product" (a comparison with the truncating version).
- `SymmetricSpace.inverse_flexural_operator` comment [history] removed "it raised on the sphere and returned a 1e-4 residual as converged on the circle"; "with_preconditioner now refuses" became "refuses". The docstring now says a solver that brings its own preconditioner keeps it.
- `SymmetricSpace._path_nodes` [v1] removed "which is v1's heuristic" (two nodes per length scale). Migration note: same default node count as v1.
- `SymmetricSpace.path_integral_operator` [covered D-12] removed "v1 computes exactly this and calls it `path_average_operator`, saying in its own docstring that the name is wrong; ... the two now differ by what they are called rather than by which version you are using."
- `SymmetricSpace.path_integral_operator` [v1] Migration note: v1 `path_average_operator` is v2 `path_integral_operator`; v2's `path_average_operator` divides by the path length.
- `SymmetricSpace._path_operator` comment [covered D-89] removed "1.59 s against 0.018 s for 2000 paths (measured 2026-08-29)" (in the appendix under D-89).
- `SymmetricSpace.geodesic_ball_average_operator` comment [stale] removed "The closed-form route below takes no point values at all."; no such route follows.
- `SpectralBlockLinearOperator` [history] "As a grid of blocks it cost ..." restated as a comparison; "fuses into products" was jargon, reworded.
- `_FlexureOperator` [history] removed "Applied term by term it cost fifty transforms per application: every `multiply` truncated back into the space" and "That held while `multiply` truncated ... and it holds now that it does not (DECISIONS.md D-87)."
- `_gauss_legendre` [history] removed "0.8 s for the 2000 calls one tomographic geometry makes (measured 2026-08-29)."

### symmetric_space/box.py

- module docstring [v1] removed "v1 builds the line from the circle and the plane from the torus in exactly this way, and for the same reason". Migration note: v1's line and plane were also padded embeddings into the circle and torus.
- module docstring [stale] removed "The support assumption is real and worth stating: a field is taken to vanish outside the domain." It was out of date: fields (and prior draws) live on the whole padded box, and only `project_function` (taper) and `support_projection` make anything vanish there. The paragraph now says the padding must be wide compared with the correlation length, and names the two methods.
- `Box.prepare_points` [history] removed the account of the bug where the NUFFT route evaluated the field displaced by the padding while the direct route was right.
- `Box.prepare_points` [RECORD] removed "it was 230 of 254 ms -- 91% -- of an application at 10^5 points (measured 2026-08-29)". Same gap as the `PreparedPoints` item in base.py: DECISIONS.md has nothing on preparing points once.
- `Box.with_order` [stale] said "the base class would give back the enclosing periodic domain". `PeriodicBox.with_order` calls `self._rebuilt`, which `Box` overrides, so it would not; reworded to state what is returned (see CODE_ISSUES).
- `Box._rebuilt` [cite D-3] "the D-3 subclass its order names" now names `Lebesgue` or `Sobolev`.
- `Box.project_function` [RECORD] removed the measurement behind `taper=True`: on the constant one over `[0, 1]` with the default padding, a path integral from 0.05 to 0.95 gives 0.851 against 0.9 (5% error) with a hard cutoff, falling to 7e-6 for a path from 0.4 to 0.6. The taper default is not in DECISIONS.md. Kept a one-line statement of the consequence.
- `Box.project_function` [v1] removed "v1 tapers for exactly this reason". Migration note: v1 also tapered across the padding.
- `Box.project_function` [history] removed "v2 stopped, and its module docstring presented the step as a 'support assumption'".
- `Box.support_projection` [cite D-85] removed "the same point as DECISIONS.md D-85".
- `Box.support_projection` [history] removed "This used to raise and tell the caller to do that by hand, which is the same operator with an extra step."
- `Interval` [v1] removed "v1's `line` space"; the docstring now says what it returns and how it relates to `line.Line`. Migration note: v1's `line` space is `symmetric_space.line.Line` (or `Interval`, which returns a plain `Box`).

### symmetric_space/line.py

- module docstring [v1] removed "v1's `line` space." Migration note: as for `Interval` above.
- module docstring [cite D-83] removed "One of the geometries of DECISIONS.md D-83"; it now says each geometry has its own module.
- `Line._rebuilt` [cite D-3] "the right D-3 subclass" now names `Lebesgue` or `Sobolev`.
- `Lebesgue.__init__`, `Sobolev.__init__` [stale] "lower, upper: the endpoints." split into two `Args:` entries, so each parameter is documented under its own name.

### symmetric_space/plane.py

- module docstring [v1] removed "v1's `plane` space". Migration note: v1's `plane` space is `symmetric_space.plane.Plane`.
- `Plane._rebuilt` [cite D-3] "the right D-3 subclass" now names `Lebesgue` or `Sobolev`.

### symmetric_space/circle.py

- module docstring [cite D-83] removed "One of the geometries of DECISIONS.md D-83".
- module docstring [cite D-84] removed "(DECISIONS.md D-84)".
- module docstring [v1] removed "where the coordinate *is* the angle and v1's numbers are these; on a circle of radius `r` a v1 angle is `r` times too small"; it now says the angle is the coordinate divided by `r`. Migration note: v1 points on a circle were angles; v2 points are arc lengths, the same only on the unit circle (D-84 records this).
- `Circle.__init__` (radius) [v1] removed "v1's parameter."
- `Circle._rebuilt` [cite D-3] "the right D-3 subclass" now names `Lebesgue` or `Sobolev`.
- `_period` [v1] "a v1 angle is this coordinate divided by the radius" became "the angle is ...".

### symmetric_space/torus.py

- module docstring [cite D-83] removed "One of the geometries of DECISIONS.md D-83".
- `Torus.__init__` (lengths) [cite D-84] removed "(DECISIONS.md D-84)".
- `Torus.__init__` (radii) [v1] removed "v1's parameters." Migration note: v1's radii (`radius_x`, `radius_y`) are `radii=`, and points are physical coordinates, not angles.
- `Torus._rebuilt` [cite D-3] "the right D-3 subclass" now names `Lebesgue` or `Sobolev`.

### symmetric_space/__init__.py

- module docstring [cite D-83] removed "See DECISIONS.md D-83 to D-91."

### symmetric_space/sphere.py

- module docstring [v1] removed "and every script written against v1" from the list of users of the `(latitude, longitude)` degrees convention. Migration note: v1 scripts already pass degrees; nothing changes for them.
- module docstring [stale] "the grid is Driscoll-Healy with the sampling given at construction" ignored the Gauss-Legendre grid; now "Driscoll-Healy by default or Gauss-Legendre on request".
- comment above `_TRANSFORM_MIN_POINTS` [cite D-89] removed "See DECISIONS.md D-89."
- same comment [history] removed the retuning story: the thresholds were 512 and 512 when finufft's default threading made the transform 4 to 20 times slower.
- same comment [RECORD] removed the measurement behind the thresholds of 200 points and dim 256: measured at lmax 4, 8, 16, 32 and 64 across six point counts, the old thresholds left a factor of 13 at lmax 64 and 1000 points (34 ms direct against 2.6 ms transform). D-89 records the NUFFT route, not the crossover thresholds.
- `_Grid` [cite D-86] removed "(DECISIONS.md D-86)".
- `_legendre_indices_for` [history] removed "Calling it per component instead made this a Python loop of length dim: 212 ms at lmax 256, against 2 ms here."
- `Sphere` [stale] "Vectors are pyshtools.SHGrid objects on a Driscoll-Healy grid" ignored GLQ; now "on the space's grid". Also tightened the paragraph on why vectors are SHGrid objects.
- `Sphere.orders` [v1] removed "v1" from "the convention v1 and pyshtools use". Migration note: the signed-order labelling is unchanged from v1.
- `Sphere.component_of` [v1] removed "v1's `index_to_integer`, vectorized". Migration note: v1 `index_to_integer(l, m)` becomes `component_of(l, m)`, now vectorized; the reverse map is indexing `degrees` and `orders`.
- `Sphere._legendre_indices` [history] removed "this was 212 ms of it at lmax 256, on every call".
- `Sphere._own_grid_values` [history] removed the account of `from_grid_values` once wrapping the caller's array with `copy=False`, so that `axpy` wrote through into it; removed "The copy it saves is 2 MB at lmax 256, on every synthesis". The docstring now says why the no-copy wrap is safe here.
- `Sphere.inner_product` [covered D-87] removed the timing "5.44 ms against 0.055 ms at lmax 128 (measured 2026-08-29)" and "Since DECISIONS.md D-87".
- same [RECORD] removed agreement figures not in DECISIONS.md: the quadrature agrees with the component route to 6e-15 at every truncation, radius and sampling tested, and to 2e-14 with one argument a raw pointwise product; when both arguments are off-span grid arrays the two routes were 0.05% to 2% apart. D-87 says the Lebesgue inner product is the grid quadrature, not why it is preferred when both arguments are off-span products (it is the more accurate of the two).
- same [history] removed "it is the same decision as 35's", which pointed to a retired journal section.
- same [stale] "the Driscoll-Healy grid comes with a quadrature" was DH-only; the code also uses the Gauss-Legendre weights. Now "the grid".
- `Sphere.truncation_degree_for` [v1] removed "The rule is v1's, so it gives v1's numbers". Migration note: `Sphere.truncation_degree_for` reproduces v1's truncation rule and numbers.
- same [stale] "sum the Sobolev weights mode by mode" in fact sums one term per degree; reworded to "degree by degree, one term per degree", which is what the following sentence ("not weighted by the 2l + 1 modes") meant.
- `Sphere.to_coefficients` [history] removed "or writing pyslfp-style interop".
- `Sphere.basis_matrix` [history] removed the profile "at lmax == 64 the Legendre evaluation itself is 3% of it, and the other 97% is indexing and trigonometry".
- `Sphere._quadrature` [history] removed the timing of one probe against one per row: "2.9 ms against 583 ms at lmax 128, and 18 ms against 8.0 s at 256".
- same [stale] the docstring described only the Driscoll-Healy calibration; the closed-form Gauss-Legendre branch is now named, and the zero pole-row weight is marked as Driscoll-Healy only.
- `Sphere._south_pole_kernel` [history] removed "It did that with a full analysis, which at lmax 256 was 19 ms of a 67 ms forward evaluation -- 35% of it (measured 2026-08-29)" and "Agrees with the analysis route to 6e-14 ...; 0.06 ms against 19 ms."
- `Sphere.prepare_points` [history] removed "Doing it per application cost 14.5 of 37 ms at 10^5 points (measured 2026-08-29)."
- `Sphere.evaluate` (nthreads) [RECORD] removed the measurement behind `nthreads=1`: on a 16-core machine finufft's default (every core) was 4 to 20 times slower than one thread, 4.4 ms against 21 ms at lmax 128 for every point count from 200 to 10000. D-52 says the NUFFT's `nthreads` is separate from `n_jobs`, but not why it defaults to one.
- `Sphere.accumulate` [stale] removed a dangling sentence fragment ("`eps` and `nthreads` mean what they do in :meth:`evaluate`, and") that ran into `Args:`; "for the reason measured there" now "given there".
- `Sphere.covariance_function` [history] removed "1.2 ms against 24 ms for 50 distances at lmax 256 (measured 2026-08-29)".
- `Sphere.walk_from` [history] removed the account of the bug where the walk did not reflect at the pole (latitude -250 at `3 pi R / 2`, the two evaluation routes disagreeing by 1.02 on a field of maximum 0.47). The docstring now says what the walk does.
- `Sphere.to_colatitude_radians` [v1] removed "and what every script written against v1 passes".
- same, comment on the latitude check [history] removed "The check the docstring has always promised and never performed" and "what would have caught walk_from returning -250 (§3.4)", which pointed to a retired journal section.
- `Sphere._to_points` [history] removed "one path of 32 nodes made 32 of these calls, and a tomographic geometry of 2000 paths made 64 000 (measured 2026-08-29)".
- `Sphere.geodesic_quadrature` comment [history] removed "Node for node the same points as the loop it replaces, to 4e-13 degrees."
- `Sphere.geodesic_ball_quadrature` [stale] Raises said "a non-positive radius"; a zero radius returns zero weights, and only a negative one raises. Corrected.
- `Sphere.cap_integral_components` [history] removed the comparison with `SHCoeffs.from_cap` (8.5 ms a center at lmax 128; 21 ms against 1437 ms over 100 centers, agreeing to 1e-12, measured 2026-08-29).
- `Sphere.geodesic_ball_average_operator` [stale] Raises "for a non-positive radius or count, or no centers" made precise: a zero radius raises only when normalizing, and a radius past half the circumference also raises.
- `Sphere.stations`, `Sphere.earthquakes` [stale] added a `Raises:` section for the refusal (raised in `_subsample`) when more points are asked for than the table holds; it was not documented.
- `Sphere.random_domain_points` [v1] removed "v1's `random_domain_points`". Migration note: same name and meaning as v1.
- `Sphere._rebuilt` [cite D-3] removed "Returns the D-3 subclass" and "That is the whole point of D-3".
- same [history] removed "a with_order/with_degree that handed back the base class defeated it on every derived space -- pyslfp's sl/utils.py dispatches on exactly that check".
- `Sphere.degree_transfer_operator` [cite D-26] removed "Getting that by hand is the mistake of DECISIONS.md D-26 wearing a different hat"; the sentence on the adjoint now says plainly that a restriction's adjoint is the matching zero-padding only when the metrics agree on the shared components.
- `_as_points` [history] removed "This used to turn them into colatitude and radians here, privately, so that a caller who read the same file themselves got different answers from the same numbers."

### symmetric_space/fourier.py

- module docstring [v1] removed "One implementation covers what v1 spreads over four files and roughly 4.1k lines" and "the three-dimensional case — the one that was not there — comes free". Migration note: v1's separate circle, torus, line and plane modules are all `PeriodicBox` (or `Box`) in v2 (D-84).
- module docstring [v1] removed "and v1's factor-of-two bookkeeping disappears" (covered by D-84, "v1's factor-of-two bookkeeping is gone").
- `PeriodicBox.component_of` [v1] removed "v1's `index_to_integer`, vectorized". Migration note: v1 `index_to_integer` on the Fourier spaces becomes `component_of(wavevector, phase=)`.
- `PeriodicBox.prepare_points` [history] removed "it was 32 of 61 ms on a 512-square torus at 10^5 points (measured 2026-08-29)".
- `PeriodicBox.to_coefficients` [stale] the docstring named the private `_component_scale` and called the method "the other half of the seam that components open"; it now states the factor `sqrt(V) / N` and that a component is the real or imaginary part of a coefficient.
- `PeriodicBox._rebuilt` [cite D-3] removed "the D-3 subclass".
- same [history] removed a stray "(measured 2026-08-29)" attached to the `isinstance` claim.
- `Lebesgue.__init__`, `Sobolev.__init__` [stale] "lengths: ... Unit lengths by default" was wrong; `PeriodicBox` defaults every period to `2 pi`. Corrected.

### plotting/sphere.py

- `subplots` for `Sphere` (projection) [covered D-92] removed "which is v1's default" and "A downstream wrapper is free to prefer another -- pyslfp defaults to Robinson -- and that is its business" (D-92 records both).
- `plot` for `Sphere` (cmap, colorbar, gridlines) [covered D-92] removed "as in v1" from each of the three defaults (`RdBu`, no colorbar, gridlines on); D-92 says the sphere's defaults are v1's. Migration note: the map defaults are unchanged from v1.
- `_rolled_to_the_dateline` [history] removed "1028 cells at lmax 128, and with them 1.2 s of the 1.3 s a map used to cost".
- `plot_points` for `Sphere` [v1] removed "which is v1's `plot_points(points, data=...)`" and "as in v1" on `cmap`. Migration note: v1 `plot_points(points, data=...)` is unchanged in v2.
- `plot_paths` for `Sphere` [history] removed the drawing timings (960 paths: 528 ms per line against 36 ms as one collection on PlateCarree, 838 against 303 on Robinson); the docstring keeps the reason (cartopy re-projects each line separately) and now says paths are split at the dateline.
- `plot_balls` for `Sphere` [stale] "with the paths' dateline splitting" and "in the units of the sphere's" were elliptical; reworded.

### plotting/fourier.py

- `plot` for `PeriodicBox` [stale] added a `Raises:` section (`ValueError` for a wrongly shaped field, `NotImplementedError` above two dimensions); both raise and neither was documented.
- `plot_error_bounds` [v1] removed "v1's `plot_error_bounds`". Migration note: same name and meaning as v1, now on any one-dimensional box.

### sem1d/__init__.py

- module docstring [cite D-3] removed "(DECISIONS.md D-3)" after "each exporting `Lebesgue` and `Sobolev`".

### sem1d/_transform.py

- module docstring [covered D-123] removed "the loop over them, in Python, was the whole cost of a draw and of every application of a covariance"; it now says such a loop would dominate the cost. Also replaced "the tests hold this to that route to rounding" with a plain statement that the result agrees with `planetmodel.harmonics` to rounding. D-123 already records the planned move to planetmodel 1.4 and this file's deletion, so the move is not logged as [RECORD].
- `ShellTransform.__init__` comment [history] "duck-typing them no longer selects" reworded so it states why the grid kind is checked by name (in planetmodel 1.3 every grid kind carries lmax and weights), without the before/after framing.

### sem1d/base.py

- module docstring [stale] "what takes its place is a variance field; see `spectral_measure`" now also points to `pointwise_variance`, which is where the variance field comes from.
- `_ROBIN_PADDING_LENGTH_SCALES` comment [covered D-122] removed "which measured does" and "(DECISIONS.md D-122)".
- `_ROBIN_COEFFICIENT` comment [covered D-122] removed "by up to an order of magnitude and never much the worse (D-122)".
- `_PADDING_LENGTH_SCALES` comment [stale] did not say it is the default under the *natural* condition only; it does now.
- `_resolved_boundary` [cite D-122] removed "(DECISIONS.md D-122)" and "measurably"; added that an inner coefficient that would be negative is held at zero, which the code does and D-122 records.
- `_sampled_with_extension` [stale] Raises said "an extension that is neither" though three are accepted; it now names all three and says that callers handle "fit" themselves (it falls through to the constant continuation here).
- `SpectralElementSpace.is_function_space` [cite D-131] removed.
- `SpectralElementSpace._class_for_order` [cite D-3] removed; it now says Lebesgue at order zero, Sobolev otherwise.
- `SpectralElementSpace.with_order` [cite D-3] removed from Returns.
- `SpectralElementSpace.multiply` [cite D-87] removed.
- `SpectralElementSpace.point_evaluation_order` [cite D-11] removed.
- `SpectralElementSpace._require_point_evaluation` [cite D-11] removed.
- `SpectralElementSpace.pointwise_variance` [cite D-26] removed "with the metric for the same reason (DECISIONS.md D-26)"; it now says why `g_k` enters (the basis is orthogonal with squared norm `g_k`, not orthonormal).

### sem1d/axis.py

- `AxisSpace.__init__` [stale] Raises did not mention the "at least one mode" or element-length refusals; added.
- `AxisSpace._coordinate_key` reworded "as the boxes' are" (ambiguous). No ledger point.
- `AxisSpace.project_function` [covered D-124] removed the measured fit errors on `cos` over `[-1, 2]` (3e-5, 3e-8, 7e-14 at 32, 64, 128 modes, coefficients no larger than the others') and "(DECISIONS.md D-124)". In the appendix ("The fills under the Robin default").
- `AxisSpace.project_function` [covered D-119] removed the continuation errors (4e-3, 1e-3, 3e-4) and "ten to twenty times the better at an end [natural condition], and under the default Robin one about two (D-119)". In D-119 and the appendix. The docstring keeps the qualitative statement that "odd" is the more accurate continuation at an end.

### sem1d/interval.py

- module docstring [cite D-122] removed "(DECISIONS.md D-122)".
- module docstring [cite D-87] and [cite D-11] removed; it now states the refusal threshold (order one half) itself.
- `Interval.__init__` (boundary) [covered D-122] removed "the point of the padding is that the condition should not matter, and measured, ... (DECISIONS.md D-122)"; the outcome stays (two length scales do what four do).
- `Interval.__init__` [stale] Raises did not mention the "at least one mode" refusal; added.

### sem1d/radial.py

- module docstring [cite D-117] removed "(DECISIONS.md D-117)"; "regular for nothing" became "regular there automatically".
- `Radial.__init__` (boundary) [covered D-122] removed "measured" and "(DECISIONS.md D-122)"; "None is applied at the centre" became "No condition is applied at the centre", since `None` here is ambiguous with `boundary=None`.
- `Radial.__init__` [stale] Raises now states the radius condition `0 <= inner_radius < radius` and the mode-count refusal.
- `Radial.point_evaluation_order` [covered D-121] removed "Measured: the norm of the Dirac's representer at the centre still grows with the modes at order 1.25 and has settled at 1.75, and at r = 0.5 has settled by 0.75 (DECISIONS.md D-121)."

### sem1d/ball.py

- module docstring [cite D-87], [cite D-2], [cite D-86] and [cite D-11] removed from the conventions paragraph.
- module docstring [covered D-117] removed "Measured, the standard deviation a prior realizes at the centre under such a profile is three per cent out at twelve radial modes and still 0.6 at forty-eight, against a hundredth of that for a profile in r^2." In the appendix. It now says that a profile linear in r converges at the centre only at first order.
- module docstring [history] removed "Targets in tests are chosen accordingly."
- module docstring [cite D-123] removed "(DECISIONS.md D-123)"; reworded "held to planetmodel's route ... and taken in its place".
- `_BATCHED_LMAX` comment [covered D-123] removed "150 times faster at degree 8, nine at 32" and "(DECISIONS.md D-123)". D-123 has 150 at 8 and 1.5 at 128, and the appendix has the per-degree timings. The comment keeps the reason for the limit (a dense Legendre step, about 1.5 times faster at 128, the largest degree measured).
- `Ball.__init__` (boundary) [covered D-122] removed "measured" and "(DECISIONS.md D-122)".
- `Ball.__init__` [stale] Raises did not mention negative `lmax` or a non-positive element length; added.
- `Ball.project_function` [covered D-124] removed "1e-9 over the domain and 2e-6 within a quarter of the radius at sixteen radial modes, where the continuations below give 5e-3 and 3e-2 (DECISIONS.md D-124)". In the appendix.

### sem1d/layered.py

- module docstring [cite D-23] and [cite D-126] removed; the paragraph now states that equality and hashing depend on the summands alone, as for any direct sum.
- `Layered._built` comment [cite D-3] removed.
- `Layered.interval`, `Layered.radial`, `Layered.ball` [stale] added the ValueError (fewer than two boundaries, a list of the wrong length) that `_built` and `_per_layer` raise.
- `Layered.with_order` reworded the Returns. No ledger point.
- `Layered.layer_index` [stale] "or the side is neither" now names the accepted values.

### backends/__init__.py

(nothing to change)

### backends/mfem.py

- module docstring [history] removed "This is the case the whole design was built for."
- module docstring [cite D-17] removed "the Gram matrix of DECISIONS.md D-17".
- module docstring [history] removed "That is deliberate: it exercises the claim that a vector can be any backend object". It now says what a user gains (MFEM's own vectors).
- `MfemSpace.is_function_space` [cite D-131] removed.
- `MfemSpace.restrict` [history] removed "This used to call `.toarray()` unconditionally, so `operator_from_bilinear_form` handed a dense array to `from_matrix` and `solver_from_bilinear_form` densified it only to re-sparsify it immediately."
- `MfemSpace.restrict_vector` [history] removed "Fancy indexing copies anyway, so only the unconstrained path needed saying — which is exactly the path where the bug is invisible until the owning form goes out of scope."
- `_to_mfem` [RECORD] (minor) removed the timing "16 ms for 33k [stored entries]" for the entry-by-entry conversion. Not in DECISIONS.md; probably not worth recording.
- `_to_mfem` [history] removed "What did matter was the dense intermediate this used to be handed", "the earlier claim that it 'has nowhere to bind' was simply wrong, and would invite someone to swap it in", "Anyone tempted by the faster constructor is being offered a segfault" and "and did". The two warnings themselves stay: the CSR constructor causes a double free, and `FormSystemMatrix` takes ownership of the form's matrix.
- `solver_from_bilinear_form` [cite D-26] removed "the error of DECISIONS.md D-26 in its most convincing disguise"; the consequence (wrong by a mass matrix) stays.
- `solver_from_bilinear_form` [stale] Raises did not mention the RuntimeError raised by the returned operator when `strict`; added.
- `functional_from_linear_form` [cite D-26] removed "the error of DECISIONS.md D-26, in the setting where it is most often made"; it now says directly that the load vector is not a gradient.
- `matern_measure` [stale] the `rtol` description had drifted into the second `Note:` section (the Args entry read "rtol, max_iterations: for the default solver"). It is now two Args entries.
- `matern_measure` [history] removed "badly, and duplicating what MFEM already does well" from the fractional-exponent note.
- `matern_measure` [stale] Raises did not mention a non-positive smoothness, the wrong number of lengths, or a rotation off two dimensions, which `_matern_parameters` and `_anisotropy` raise; added.
- `_anisotropy` [stale] claimed `correlation_length` is "the distance at which the correlation has fallen to about 0.14". Computed from the Matérn kernel with `kappa = sqrt(2 nu) / l`: the correlation is about 0.14 at *twice* the correlation length, for every smoothness, and between 0.37 (nu = 1/2) and 0.61 (nu → ∞) at the correlation length itself. Corrected.

### backends/mfem_hilbert.py

- module docstring [history] removed "the test environment has no `mpi4py`, so the claim is that the design admits it, not that it has been run". The caveat stays: the MPI path is untested.
- module docstring [cite D-85] removed "DECISIONS.md D-85 describes exactly this chain; a finite element space is its most natural instance." D-85 is about Sobolev spaces being diagonal-metric and not mass-weighted, so the citation was also misdirected.
- `MfemDofSpace` [cite D-26] removed "the thing DECISIONS.md D-26 warns against mistaking for an inner product of functions".
- `MfemHilbertSpace.is_function_space` [cite D-131] removed.
- `solver_from_bilinear_form` [cite D-26] removed "DECISIONS.md D-26 in its most convincing disguise".
- `solver_from_bilinear_form` [stale] added a Raises section for the RuntimeError on non-convergence under `strict`.
- `white_noise_load` comment [stale] "see the coordinate backend for why" replaced by the reason itself, so the comment stands alone.
- `matern_measure` [stale] Raises said "(see the note in the source)"; it now states the PyMFEM 4.8 partial-assembly fault and the other refusals of `_matern_parameters`.
- `matern_measure` comment [covered D-97] removed "(measured; ...)" and kept the fact: a matrix coefficient under partial assembly gives values of order 1e290, while a scalar one agrees with full assembly.

### plotting/sem1d.py

- module docstring [cite D-87] removed; an over-long first line was also reflowed.
- `plot_shell`, `plot_section`, `plot_profile`, `plot_section_points` [stale] `space` was documented as "the ball or annulus" though all four take a `Layered` space of balls; corrected. Raises now lists the TypeError for non-ball layers, the interface and gap errors, the coincident-points error of `through=`, and `padding` refused on a layered space (raised in `plot_section`'s own body).
- `_section_frame` [stale] Raises said "unless exactly one of the three names the plane"; none is allowed too (the meridian of longitude zero). Corrected.

### examples/03_operators.py

- module docstring [v1] "There are no dual spaces in pygeoinf 2" is now "There are no dual spaces here". Migration note: v1's dual spaces (`to_dual`, `from_dual`, `DualHilbertSpace`, `LinearForm`) have no counterpart; the adjoint is the only map (D-16).

### examples/10_measures.py

- comment before `log_density` [cite D-131] removed.

### examples/15_worked_example.py

- comment above `prior_error` [history] removed "Written as the comparison rather than as the 1.0 it currently equals, so it stays correct if the prior ever gets a non-zero mean"; it now says the prior mean is zero, so the value is 1.0.

### examples/16_mfem_backend.py

- comment on the check report [cite D-128] removed.
- comment above `stiffness_matrix` [history] removed the reviewer's note that "an example that reaches into a module's internals teaches the wrong thing" (the example used to call the backend's private converter).

### examples/17_sets.py

- module docstring [v1] removed "v1 keeps these in three places and nothing connects them". Migration note: v1 kept a set's membership, indicator and support function as separate objects; v2's `ConvexSet` carries all three (D-67).

### examples/20_flexure.py

- module docstring [v1] removed "v1's `work/flexure.py` on v2". Migration note: this example is the port of `work/flexure.py`.
- module docstring [history] "Three things had to exist for it to be written at all" is now "It rests on three things".

### examples/21_tomography.py

- module docstring [v1] removed "v1's `work/tomo.py` on v2". Migration note: this example is the port of `work/tomo.py`.
- module docstring [stale] said the normal equations "were assembled in whichever space is smaller, without being asked". The formalism defaults to `"data_space"` regardless of size (D-74), as the example's own later comment says; corrected.
- comment above the posterior [stale] said "nothing here is ever assembled", but the forward operator is built with `dense=True`; it now says the normal operator is applied, never assembled.

### examples/22_coupled_fields.py

- module docstring [v1] removed "v1's `work/dynamic_topography.py` on v2". Migration note: this example is the port of `work/dynamic_topography.py`.
- module docstring [stale] "is the same call it was before" had no antecedent; it is now "as in example 21".
- comment on `grid_values` [cite D-1] removed.
- comment above the solver [stale] "nothing here is ever assembled" narrowed to the normal operator.

### examples/23_feasible_set.py

- module docstring [v1] removed "v1's `work/sphere_dli_example.py` on v2". Migration note: this example is the port of `work/sphere_dli_example.py`.
- module docstring and section comments [stale] "Three routes compute it, and the example runs two", with sections headed "Route (c)" and "Route (b)", referred to a labelling that is not defined anywhere the reader can see (there is no route (a)). Now: the exact bounds, and a linear certificate bounding them from outside.
- comment above `certificate` [stale] removed "which is what BGP recommends anyway": BGP is not defined in the example.

### examples/24_preconditioning.py

- module docstring [stale] said example 21's one-line inversion was "fine while the normal equations fit in a Cholesky factorization. When they do not, the solve becomes iterative". Example 21 uses the default solver, which is CG (D-39); corrected.

### examples/25_distributions.py

- comment above `estimator` [covered D-39] removed "That is section 27.5's criterion", a citation of a section of the retired design journal. The criterion (choose the solver by how often the inverse is applied) is D-39's.
- same comment [stale] removed "this is the only place in the examples where a direct solver is the right default": examples 22, 23, 24 and 26 also choose `CholeskySolver`.
- comment above `summary` [cite D-73] removed.

### examples/27_mfem_inverse.py

- module docstring [cite D-26] "the distinction of DECISIONS.md D-26" is now "the distinction of example 5".
- module docstring [stale] "This library never sees a matrix" is false for the coordinate backend, which reads the mass matrix and the stiffness form's sparse matrix. Narrowed to "never inverts a matrix", which is the claim being made about the PDE solve.

### examples/35_evaluation_failure.py

- module docstring [cite D-129] removed.

### tests (decision citations only)

In the tests, only citations and planning references were removed. The regression history in test docstrings stays, since it records why each test exists. Each sentence was reworded to state the point instead of the number.

- `test_absolutely_continuous.py` module docstring [cite D-132]; `test_the_algebra_degrades_to_a_plain_gaussian` [cite D-73] ("D-73's worry, answered").
- `test_escape_hatches.py` module docstring [cite D-88]: "dropped on purpose; DECISIONS.md D-88 says why" now gives the reason, that no problem reaches the scale where the pooled nodes do not fit; [history] "the escape hatches the audit found closed".
- `test_bounded_and_sphere.py` [cite D-2], [cite D-1], [cite D-3], one each in three docstrings.
- `test_code_practice.py` comment on `CONVENTIONAL` [cite D-52]; `TestDocstringsCarryTheContract` [history] removed "K Should-12", a review-item label; `TestTheNamespaceHasNoHoles` [cite D-5].
- `test_mixture.py` module docstring [cite D-66].
- `test_compat.py` `TestWhiteNoiseIsFixed` [cite D-19], now says what the v1 defect was; `TestDiracAndRepresenters` [cite D-26]; a fixture comment [cite D-19].
- `test_observation.py` `TestPointConvention` [cite D-2].
- `test_fourier_spaces.py` [cite D-85] twice; [cite D-26]; [stale] "DECISIONS.md D-85 and 13.2", where 13.2 is a section of the retired design journal.
- `test_direct_sum.py` `TestJointModel` [cite D-23]; the docstring now names the construction.
- `test_operators.py` check-report test [cite D-128].
- `test_sem1d_plotting.py`, `test_sem1d_interval.py`, `test_sem1d_radial.py`, `test_sem1d_layered.py`, `test_sem1d_ball.py` [cite D-107] in each module docstring; `test_sem1d_interval.py` [cite D-122]; `test_sem1d_ball.py` [cite D-124] twice and [cite D-119].
- `test_convex.py` `test_its_hessian_is_the_identity` [cite D-26].
- `test_normal.py` module docstring [cite D-44].
- `test_laplace.py` module docstring and `TestTheSamplerHooksAreThere` [cite D-7].
- `test_backus.py` module docstring and two test docstrings [stale] "Routes (a), (b) and (c) of DECISIONS.md D-79": D-79 no longer names its routes by letter. They are now named: the closed form, the linear certificate, the primal bisection.
- `test_nonlinear.py` [cite D-26].
- `test_inference.py` module docstring [cite D-71].
- `test_optimization.py` [cite D-26] (now: "why the optimizers are written in the space rather than wrapped around SciPy"); [cite D-54]; [cite D-129].
- `test_point.py` module docstring [cite D-57] and [cite D-76]; the preconditioner-in-the-sweep test [cite D-44].
- `test_probability.py` `TestWhiteNoiseCorrection` [cite D-19]; `TestParallelLoops` [cite D-6]; `TestTheFunctionSpaceMarkerGatesDensities` [cite D-131] and [cite D-11].
- `test_space_parity.py` `TestChangingTheMetricKeepsTheSubclass` [cite D-3] twice; the order-guard test [cite D-11]; two power-measure tests [cite D-130].
- `test_sphere_transform.py` module docstring [cite D-89]; [cite D-87]; [cite D-86].
- `test_spectral.py` prior-weighted range test [cite D-53]; [history] removed "David remembered this working far better on function spaces".
- `test_spaces.py` `TestRepresenter` [cite D-26]; `TestWhiteNoise` [cite D-19], whose v1 remark is now a statement about standard normal components.
- `test_phase_four.py` `TestHardening` [history] removed "(§18.1)", a section of the retired design journal.
- `test_backus.py` `test_the_inner_hull_sits_inside_the_outer_bound` [history] "§18.4's sandwich, with both sides real for the first time" now says what the sandwich is.
- `test_code_practice.py` `test_the_quadratic_form_module_does_not_import_scipy_at_module_scope` [stale] said `inference.problem` "still imports `scipy.stats` eagerly"; it imports it lazily now. Reworded to the reason that still holds.
