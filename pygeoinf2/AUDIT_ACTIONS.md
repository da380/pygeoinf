# Acting on the functionality audit

One line per item from `FUNCTIONALITY_AUDIT.md` §0.2–§0.4, worked through one
at a time: one item, one session, one commit. Each line ends in a verdict —
**restored**, **subsumed** (how), **dropped** (why), or **not a defect** — with
the commit that settled it. Pick the next unticked line rather than re-reading
the audit; the per-symbol evidence is in its §2–§3 tables.

Baseline on 2026-09-16: 2454 passing in the fast suite.

## 1. Correctness claims (§0.3, "changed numerics")

- [x] `with_regularized_inverse`: covariance and precision disagree for `damping > 0` — **dropped**. The mismatch was real (the mixture density it fed raised), but its one use was a damped `Q⁻¹` for the Woodbury data form, which is now `WoodburyPreconditioner(prior_damping=)`. DESIGN §36.
- [x] `WoodburyPreconditioner.data_form()` on a Tikhonov operator returns `t·N⁻¹` not `N⁻¹` — **restored**, and wider than the audit saw: every factor-built preconditioner had the factor. `FactoredNormalOperator.scale` now carries it and each consumer divides by it. DESIGN §37.
- [x] adjoint solve of a non-self-adjoint operator with a fixed preconditioner uses `P` not `P*` — **restored**: iterative inverses now carry an adjoint solve preconditioned by `P*` from the same resolved `P`, which also ends the second factorisation of a deferred direct preconditioner. Right answer, slow, before; never a wrong number. DESIGN §38.
- [x] `operator_log` has no floor on Ritz values — **restored**, generalised: both Lanczos kernels now hold Ritz values to the claimed spectrum (floor at 0 for semidefinite, `eps·λmax` for definite), and refuse a meaningfully negative one as a false claim. Triggered by formed `L L*` covariances, which also broke `sqrt` and fractional powers and every stochastic log-determinant. DESIGN §39.
- [x] CG lost its non-finite breakdown checks — **restored**, and for every iterative solver: a non-finite residual is refused at once in the shared per-step hook, regardless of `strict`; CG also refuses `(r, P r) <= 0`. Before, a NaN ran to the iteration cap (800 applications at dim 400) and under `strict=False` came back as a NaN answer. DESIGN §40.
- [x] `apply_operator_function` caps at 50 iterations with `rtol=1e-10` — **restored** v1's dimension cap and kept v2's tolerance: measured, the cap of 50 never let the tolerance be met (50 applications always; 1e-3 to 1e-1 error on hard spectra, silently), v1's `1e-3` delivered 1e-2, and `1e-8` delivered 1e-6. Same for the quadratic form and `OperatorFunction`. DESIGN §41.
- [x] path operators bypass the Sobolev-order guard — **restored**, with the right threshold: a path integral needs order above `(d-1)/2` (one half on a surface, measured: the representer diverges at and below it), not point evaluation's `d/2` that v1 used; `unsafe=True` on both path methods. Ball averages need no order and keep their bypass. DESIGN §42.
- [x] `StrongWolfeLineSearch._zoom` is pure bisection — **restored**: cubic/quadratic interpolation with bisection as the safeguard, as SciPy's search that v1 wrapped; the wasted re-evaluation on zoom entry is gone. Nonlinear CG halves its evaluations (224→113 Rosenbrock, 4321→2013 on a cond-1e4 quadratic), same answers. DESIGN §43.
- [x] `LevelBundleMethod`: serious/null step test, QP warm start, λ bounding box — **restored** all three, plus `lower_bound` and `serious_steps` on the result. DESIGN §44.
- [x] `ProximalBundleMethod`: exact QP backend replaced by projected gradient with a residual floor — **restored**: the k-variable dual now goes through Clarabel or OSQP when installed (30× faster, 1000× closer to the primal on the Backus dual), projected gradient as the fallback; backend order now Clarabel first, OSQP having capped out on 40% of level masters. DESIGN §44.
- [x] v1's probed default adjoint versus v2's `NotImplementedError` — **subsumed** as an opt-in: `with_probed_adjoint()` assembles the matrix once and derives the adjoint from it; the refusal stays the default and its message names the opt-in. DESIGN §45.

## 2. Silent unit changes (§0.3)

- [x] circle/torus `geodesic_distance` and `project_function` take physical coordinates, not angles — **kept** (every box is physical), but the named classes now default to the unit circle and torus (period `2π`, as the generic box and v1's radius one already did) and take `radius=`/`radii=`; on a unit circle the coordinate is the angle and v1's numbers are v2's. DESIGN §46.
- [x] `degree_multiplicity` at the Nyquist degree — **not a defect**: one is right, v1's two was the bug Dan's fix branch corrects. DESIGN §46.

## 3. Dense-by-default regressions (§0.3), in the audit's order of likely pain

Moved ahead of the lost capabilities on 2026-09-16: these bear directly on the
matrix-free aim and the audit ranks the first six as the ones most likely to
bite on a real problem; the small restorations below matter less in practice.

- [x] `with_sparse_approximation` forms the dense covariance first — **restored** as an operator: `numerics.sparse_approximation` probes columns matrix-free with v1's correlation criterion and cap, sharing the thresholded preconditioner's assembly; `GaussianMeasure.sparse_covariance` delegates; the measure-returning method is gone. DESIGN §47.
- [x] `nuclear_norm` / `hilbert_schmidt_norm` default dense; correlated invariant measures assemble the block matrix — **restored**: `auto` is exact and matrix-free (spectrum, spectral slices, stored matrix, or probed diagonal in linear memory); `dense` is opt-in. KL's refusal above `dense_limit` kept as D-8. DESIGN §48.
- [x] `credible_set` without a precision goes O(N³); `ambient_ball(method='auto')` picks dense `eigh` — credible set **restored** matrix-free: the precision is the covariance's inverse through a solver (CG by default, a direct solver by name); ambient ball **kept**, the dense route being cheaper than sampling at its limit and exact. DESIGN §49.
- [x] `FeasibleProperty.is_feasible` does a dense `eigh` — **restored** v1's matrix-free test: a damped minimum-norm root search in the data space, one warm-started Krylov solve per probe; the dual route uses it too when its sets are balls. Data spaces are small only relative to model spaces. DESIGN §50.
- [x] `l2_products_operator` stacks a dense matrix; low-rank factors stored as dense blocks — **not a defect** as stated (the rows and the column blocks are the vectors themselves, and smaller than the fields), but a real trap beside it **fixed**: a composition's known matrix multiplied low-rank factors into a dense n×n product for any caller that asked; it now declines a product larger than its largest factor and gives the diagonal of a low-rank product in O(nk). DESIGN §51.
- [x] `LinearGaussianInversion` factorises at construction; `normal_log_determinant` forms dense Galerkin matrices — construction **restored** lazy at the solver: every direct solver now factorises on first use and keeps the factors (v1 refactorised per call); the log-determinant's routing **kept**, it already counts applications. DESIGN §52.
- [x] `Ellipsoid.project` factorises the dense Galerkin matrix every Newton step — **fixed** (new in v2, no v1 path): the two solves per step go through `resolve_solver`, CG at 1e-12 by default with a first-order predictor as warm start; a direct solver by name as before. DESIGN §53.
- [x] `DiagonalMetricSpace.gram_matrix()` probes a dense array; `MassWeightedSpace.mass_inverse` defaults to CG — **restored**: the Gram matrix is `diag(metric_values)` written down, the dense log-determinant sums the diagonal's logs instead of a cubic `slogdet`, and a diagonal mass inverts exactly as its docstring promised. DESIGN §54. Group 3 closed.

## 4. Lost capabilities with an obvious home (§0.2)

- [x] 4. `SolutionTrackingCallback`: the solver callback cannot see the iterate — **restored**: every iterative solver hands its callback a `SolveStep` whose `iterate` is formed on demand (free where the solver holds it, assembled from the Arnoldi basis in GMRES only when asked) and returned as a copy; `SolutionTrackingCallback` keeps them in `iterates`. DESIGN §55.
- [x] 3. `HalfSpace` support function raises — **restored**: `SupportFunction.of_half_space` is extended-real valued (`alpha * offset` along the outward normal, `+inf` elsewhere, parallelism decided on the residual in the space's norm), its maximiser the boundary's least-norm point, `ValueError` in an unbounded direction; the hyperplane gets the two-sided version. DESIGN §56.
- [ ] 2. `SublevelSet` / `LevelSet`, `is_empty`, `is_bounded`, `closure`, `boundary`, convexity `check`
- [ ] 17. `BackusInference` is a name reused for a different route
- [ ] 16. `CallableSupportFunction.support_point`
- [ ] 14. affine-operator axiom checks in `testing.py`
- [ ] 12. public spectral indexing (`indices`, `index_to_integer`, …)
- [ ] 13. saddlepoint `weighted_chi2_cdf`, array thresholds, finite-difference dual gradient
- [ ] 18. invariant-measure algebra loses its Karhunen–Loève sampler
- [ ] 5. `weakened_ellipsoid`, Cameron–Martin credible set, `sample_pointwise_variance`, KKT push-forward precision
- [ ] 10. `LinearOperator.matrix(dense=False)` scipy bridge
- [ ] 6. `random_domain_points`, the `extend` grid option
- [ ] primal feasible-property route is norm balls only: a non-identity error covariance is a whitening of the data by its factor (Mag's discussion with David); the feasibility search already takes any misfit (`point.misfit_search`, DESIGN §50), the data-space reduction does not

## 5. Lost knobs (§0.3)

- [ ] adaptivity: `rtol=` through `random_diagonal`, `deflated_diagonal`, `JacobiPreconditioner`, `SpectralPreconditioner`
- [ ] `random_range`: `measure=` probes and `power` default 2 → 1; `random_cholesky` Nyström; `random_trace` probe type
- [ ] parallelism: `n_jobs=` on point/path evaluation, preconditioner probing, `as_multivariate_normal`, log-determinant, `support_values`
- [ ] escape hatches: `lazy_quadrature`, `incomplete=True`, `inverse_sqrt_operator` route, `to_coefficient_operator` padding, degree proposal beyond the space
- [ ] a derivative-only query on any `Operator` goes through `at()` and so evaluates the value too (v1 called the derivative closure alone): doubles a derivative-only query on a forward operator whose value is a PDE solve; line searches take that path. Audit row on `NonLinearOperator.derivative`
- [ ] `DiscrepancyPrinciple` dropped v1's `atol` and `minimum_damping`; the root finder has both but does not expose them

## 6. Naming (§0.4) — one pass, last, with `compat` carrying the old names

- [ ] decide each row of the §0.4 table
- [ ] apply in one commit
- [ ] settle `maxiter` / `max_iterations` / `iterations`

## 7. Big-ticket items needing a decision, not a session

- [ ] 1. `SubspaceSlicePlotter` / `plot_slice`
- [ ] 9. point, geodesic and network plotting on boxes/tori/planes; source/receiver markers
- [ ] 7. dataset downloaders and the cache directory
- [ ] 11/15. `MassWeightedHilbertModule`; `MassWeightedSpace` as a `CoordinateSpace`; `EuclideanSpace.subspace_projection`
- [ ] 8. `dynamical_system.py`
