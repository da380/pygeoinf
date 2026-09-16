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
- [ ] `apply_operator_function` caps at 50 iterations with `rtol=1e-10`
- [ ] path operators bypass the Sobolev-order guard
- [ ] `StrongWolfeLineSearch._zoom` is pure bisection
- [ ] `LevelBundleMethod`: serious/null step test, QP warm start, λ bounding box
- [ ] `ProximalBundleMethod`: exact QP backend replaced by projected gradient with a residual floor
- [ ] v1's probed default adjoint versus v2's `NotImplementedError`

## 2. Silent unit changes (§0.3)

- [ ] circle/torus `geodesic_distance` and `project_function` take physical coordinates, not angles
- [ ] `degree_multiplicity` at the Nyquist degree (1 in v2, 2 in v1; Dan's v1 fix branch says 1 is right)

## 3. Lost capabilities with an obvious home (§0.2)

- [ ] 4. `SolutionTrackingCallback`: the solver callback cannot see the iterate
- [ ] 3. `HalfSpace` support function raises
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

## 4. Dense-by-default regressions (§0.3), in the audit's order of likely pain

- [ ] `with_sparse_approximation` forms the dense covariance first
- [ ] `nuclear_norm` / `hilbert_schmidt_norm` default dense; correlated invariant measures assemble the block matrix
- [ ] `credible_set` without a precision goes O(N³); `ambient_ball(method='auto')` picks dense `eigh`
- [ ] `FeasibleProperty.is_feasible` does a dense `eigh`
- [ ] `l2_products_operator` stacks a dense matrix; low-rank factors stored as dense blocks
- [ ] `LinearGaussianInversion` factorises at construction; `normal_log_determinant` forms dense Galerkin matrices
- [ ] `Ellipsoid.project` factorises the dense Galerkin matrix every Newton step
- [ ] `DiagonalMetricSpace.gram_matrix()` probes a dense array; `MassWeightedSpace.mass_inverse` defaults to CG

## 5. Lost knobs (§0.3)

- [ ] adaptivity: `rtol=` through `random_diagonal`, `deflated_diagonal`, `JacobiPreconditioner`, `SpectralPreconditioner`
- [ ] `random_range`: `measure=` probes and `power` default 2 → 1; `random_cholesky` Nyström; `random_trace` probe type
- [ ] parallelism: `n_jobs=` on point/path evaluation, preconditioner probing, `as_multivariate_normal`, log-determinant, `support_values`
- [ ] escape hatches: `lazy_quadrature`, `incomplete=True`, `inverse_sqrt_operator` route, `to_coefficient_operator` padding, degree proposal beyond the space

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
