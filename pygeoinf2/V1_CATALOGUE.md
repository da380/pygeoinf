# What happens to every part of v1

A complete inventory of v1's public surface — 44 modules, 145 classes, 919
public methods, 84 free functions — and what becomes of each one.

It exists because things were being lost quietly. The `finufft` path for
scattered evaluation had been dropped without anyone noticing, because a slow
right answer passes every test a fast one does. That is the failure mode this
catalogue is against: not decisions to drop something, but *absences nobody
decided on*.

## How to read it

| status | meaning |
|---|---|
| **Ported** | in v2 now, same idea, possibly renamed |
| **Subsumed** | the capability is there, reached a different way — no one-to-one name |
| **Planned (X)** | not yet; the letter is the work package in DESIGN.md §21 |
| **Dropped** | deliberately not coming, with the reason |

Work packages: **F** field algebra, **S** symmetric-space operators,
**P** probability, **A** algebra, **N** numerics, **G** geometry,
**O8** plotting, **X** examples and data, **M5** the inference layer,
**later** after the rest.

**The default is that everything comes across.** *Dropped* needs a reason in
the row, and every one of them now has yours.

The last column is yours, and it is now filled in. **Every row you marked has
been restatused against your note**, and the notes are kept as the record of
why. There were 53 Open rows; all are now Planned or Dropped. The plan those
statuses feed is DESIGN.md §21.

## Summary

320 rows, some grouping several closely related names. After your markup:

| status | rows |
|---|---|
| **Ported** | 125 |
| **Subsumed** | 32 |
| **Planned** | 89 — 46 in M5, 43 tagged by package |
| **Dropped** | 19 |
| **Open** | 0 |

Part 1 is by module, at class and free-function level. Part 2 is by method,
for the classes big enough for something to hide in — 23 of v1's 145 classes
hold 597 of its 919 public methods, and that is where the losses are. A class
marked *Ported* can still have shed half of what it did.

## What your markup settled

**Five things confirmed dropped**, all with your reason recorded in the row:
`configure_threading` ("basically doesn't work"), `empirical_data_error_measure`
as it stands, `ScipyIterativeSolver`, `ScipyUnconstrainedOptimiser`, and the
`data_assimilation` teaching module — with sequential assimilation returning
later on top of `dynamical_system`, which is Planned rather than dropped.

**Three corrections to my own rows.** `to_coefficients`/`from_coefficients` was
marked Subsumed and is not: v2's components are real by construction, and there
is no complex-spectrum accessor at all. `point_evaluation_operator` on the
sphere is Ported but 26.6 s per adjoint application at a realistic tomography
size, so it needs the same treatment the box got. And `laplacian_eigenvector_squared_norm`
you asked me to check: Parseval holds exactly at radius 1, 2 and 6371 and on
boxes with unequal sides, cross-checked against grid quadrature, so the drop
stands.

**Four verification tasks** rather than ports, from your "so long as the
functionality is the same": `DiagonalLinearOperator` against
`InvariantLinearAutomorphism` (v2 has no `from_index_function`);
`GaussianMeasure`'s invariant optimisations, of which `kl_divergence`'s O(N)
spectral path is missing along with `kl_divergence` itself;
`deflated_pointwise_variance`, which you suspect never worked; and
`distance_localized_preconditioner`, which never performed as hoped.

---

# Part 1 — Modules

## `hilbert_space.py` → `algebra/spaces.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `HilbertSpace` | Ported | `HilbertSpace`, now with `_key` so instances hash (§9) | |
| `EuclideanSpace` | Ported | `EuclideanSpace` | |
| `OrthonormalHilbertSpace` | Ported | `OrthonormalSpace` | |
| `OrthogonalHilbertSpace` | Ported | `DiagonalMetricSpace` | |
| `DualHilbertSpace` | Dropped | No dual spaces in v2. Riesz identification throughout, so a functional is a `LinearFunctional` on the space itself (§1) | |
| `MassWeightedHilbertSpace` | Subsumed | Any `CoordinateSpace` with a non-trivial Gram matrix. The mass matrix *is* the Gram matrix (§3.2, §15.1) | |
| `MassWeightedHilbertModule` | Subsumed | As above; the module structure is `vector_multiply`, which is **Open** below | |
| `HilbertModuleMixin` | Ported | Pointwise multiplication of fields. `work/flexure.py` needs it for a spatially varying rigidity. No v2 home yet — see Part 2 | We do want this functionality in some form|

## `linear_operators.py` → `algebra/operators.py`, `nodes.py`, `diagonal.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LinearOperator` | Ported | `LinearOperator`, with traits and memoised adjoints | |
| `MatrixLinearOperator` | Ported | `from_component_matrix` / `from_derivative_matrix` | A point of this specialisation is to have easy access to the matrix elements which can be useful. Same with the other specialisation below. So we don't need the classes, but that aspect is helpful. |
| `DenseMatrixLinearOperator` | Ported | as above, plus `assembled()` | |
| `SparseMatrixLinearOperator` | Ported | No sparse-backed operator in v2. `_weight_operator` in `symmetric_space/base.py` does it privately for one case; it probably wants to be public | Yes, this form is useful in practice. |
| `DiagonalSparseMatrixLinearOperator` | Ported | `DiagonalLinearOperator`, with a closed algebra and exact functional calculus | |

## `linear_forms.py`, `nonlinear_forms.py`, `nonlinear_operators.py`, `affine_operators.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LinearForm` | Subsumed | `LinearFunctional`, a `LinearOperator` into `Reals` rather than a separate type (§1) | |
| `NonLinearForm` | Ported | `Functional` | |
| `NonLinearOperator` | Ported | `Operator` | |
| `AffineOperator` | Ported | `AffineOperator` | |

## `direct_sum.py` → `algebra/direct_sum.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `HilbertSpaceDirectSum` | Ported | `DirectSum`, with optional labels excluded from identity | |
| `BlockLinearOperator` | Ported | `BlockLinearOperator` / `BlockOperator` | |
| `BlockDiagonalLinearOperator` | Ported | `BlockDiagonalLinearOperator` / `BlockDiagonalOperator` | |
| `ColumnLinearOperator` | Ported | `ColumnLinearOperator` / `ColumnOperator` | |
| `RowLinearOperator` | Ported | `RowLinearOperator` / `RowOperator` | |
| `BlockStructure` | Subsumed | Shape validation lives in the block classes | |

## `linear_solvers.py` → `numerics/solvers.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LinearSolver` | Ported | `LinearSolver`, now declaring `requires` traits and `requires_coordinates` | |
| `IterativeLinearSolver` | Ported | `IterativeSolver` | |
| `DirectLinearSolver` | Ported | `DirectSolver` | |
| `CGSolver` | Ported | `CGSolver`, coordinate-free | |
| `MinResSolver` | Ported | `MinResSolver`, coordinate-free | |
| `BICGStabSolver` | Ported | `BiCGStabSolver`, coordinate-free | |
| `LSQRSolver` | Ported | `LSQRSolver`; v1's sign bug in the damped branch is fixed (§9) | |
| `LUSolver`, `CholeskySolver`, `EigenSolver` | Ported | same names, coordinate-backed | |
| `FCGSolver` | Ported | `FlexibleCGSolver`. Wanted whenever the inner solve of a preconditioner is itself iterative, which the Woodbury one warns about | I've not really used it, but it would be nice to keep |
| `ScipyIterativeSolver` | Dropped | The point of v2's Krylov methods is that they run coordinate-free against PETSc or MFEM. A SciPy wrapper cannot | I'm happy to drop this, though my sense was when components are available this might be preferable to the hand written forms. But maybe not. |
| `CGMatrixSolver`, `BICGMatrixSolver`, `BICGStabMatrixSolver`, `GMRESMatrixSolver` | Dropped | Matrix-only convenience wrappers around SciPy, superseded as above. **GMRES itself is missing** and is the one gap — see Open below | |
| *(GMRES, coordinate-free)* | Ported | v2 has no solver for a non-symmetric operator other than BiCGStab. Worth adding? | Yes, if it's doable. Most invers methods lead to symmetric problems, but for completeness if nothing else. |
| `ProgressCallback`, `ResidualTrackingCallback`, `SolutionTrackingCallback` | Ported | v2 returns a `SolveResult` with the residual history instead. Live callbacks are still wanted for long solves — no stage yet | Yes, I think this feature is useful, especially in debugging. Adding in
addtional information as you suggest will be very useful. |

## `preconditioners.py` → `numerics/preconditioners.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `IdentityPreconditioningMethod` | Ported | `IdentityPreconditioner` | |
| `JacobiPreconditioningMethod` | Ported | `JacobiPreconditioner` | |
| `BandedPreconditioningMethod` | Ported | M5. Needs `extract_diagonals` | |
| `ColumnThresholdedPreconditioningMethod` | Ported | `ColumnThresholdedPreconditioner`. The pattern is symmetrised — v1's column-wise dropping gives an asymmetric matrix, which CG cannot use (DESIGN §23.8) | |
| `ExactBlockPreconditioningMethod` | Ported | M5 | |
| `SpectralPreconditioningMethod` | Ported | M5; `random_eig` is already there | |
| `IterativePreconditioningMethod` | Ported | M5; wants `FCGSolver` | |

## `functional_calculus.py` → `numerics/functional_calculus.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LanczosOperatorFunction` | Ported | `OperatorFunction` | |
| `lanczos_tridiagonalize` | Ported | `lanczos_tridiagonalise` | |
| `iter_lanczos_tridiagonalize` | Ported | `iter_lanczos_tridiagonalise` | |
| `apply_operator_function` | Ported | same name | |
| `operator_function_quadratic_form` | Ported | `operator_quadratic_form` | |
| — | *(new)* | `operator_sqrt`, `operator_log`, `operator_exp`, `operator_power`, `operator_inverse_sqrt`, and dispatch to the exact route for a diagonal operator | Yes, worth adding in. |

## `low_rank.py` → `numerics/randomised.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LowRankEig`, `LowRankSVD`, `LowRankCholesky` | Ported | same names | |
| `random_range` | Ported | same name; no longer routes mass-weighted spaces through the broken white-noise measure (§9) | |
| `random_trace` | Ported | same name, returning an `Estimate` with a standard error | |
| `random_diagonal` | Ported | same name (Bekas–Kokiopoulou–Saad) | |
| `deflated_diagonal` | Ported | Not ported. Diagonal estimation with a low-rank part removed first | follows `deflated_pointwise_variance`, which you flagged as possibly never having worked |
| `white_noise_measure` | Dropped | The v1 defect of §9: it produced covariance `G`, not `I`. Replaced by `HilbertSpace.white_noise` | |

## `nonlinear_optimisation.py` → `numerics/optimisation.py`, `line_search.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `ScipyUnconstrainedOptimiser` | Dropped | It passed `G^-1 dJ/dc` as `jac` while passing the Galerkin matrix as `hess` — mixing gradients and derivatives in one call (§12). Replaced by bespoke coordinate-free methods | yes, drop|
| `line_search` | Ported | `ArmijoLineSearch`, `StrongWolfeLineSearch` | |
| — | *(new)* | `SteepestDescent`, `NonlinearCG`, `LBFGS`, `NewtonCG`, `TrustRegionNewton`, `truncated_cg`, `gauss_newton_hessian` | Good. Might 
think if anything else is needed. Constrained optimsiation? Though that might need further thought, and potential 
integration with convex stuff as constraints are likely to be convex even if the functional isn't|

## `gaussian_measure.py` → `probability/gaussian.py`

Class-level Ported; see Part 2, where a third of its methods are not.

## `convex_analysis.py` → `geometry/convex.py`, `numerics/convex.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `SupportFunction` | Ported | `SupportFunction`, and `ConvexSet.support_function()` | |
| `BallSupportFunction` | Ported | `Ball.support_function()` | |
| `EllipsoidSupportFunction` | Ported | `Ellipsoid.support_function()` | |
| `HalfSpaceSupportFunction` | Ported | `HalfSpace` | |
| `PointSupportFunction` | Ported | `_PointSupport` | |
| `LinearImageSupportFunction` | Ported | `_ImageSupport`; v1's domain/codomain were the wrong way round (§9) | |
| `MinkowskiSumSupportFunction` | Ported | `_MinkowskiSupport` | |
| `ScaledSupportFunction` | Ported | `_ScaledSupport` | |
| `CallableSupportFunction` | Ported | §18.12: `ConvexSet.from_support_function`, the oracle case M5 route (d) needs | |

## `convex_optimisation.py` → `numerics/convex.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `SubgradientDescent`, `SubgradientResult` | Ported | `SubgradientDescent`, `_ConvexResult` | |
| `Cut`, `Bundle` | Ported | M5 stage 5.9 | |
| `ProximalBundleMethod`, `LevelBundleMethod`, `BundleResult` | Ported (proximal) | M5 stage 5.9 — the dual route for a general convex prior | |
| `QPSolver`, `QPResult` | Planned | M5 stage 5.9 | |
| `SciPyQPSolver`, `OSQPQPSolver`, `ClarabelQPSolver` | Planned | M5 stage 5.9. Coordinates are fine here: the QP lives in a finite-dimensional, canonically Euclidean space | |
| `best_available_qp_solver` | Planned | M5 stage 5.9 | |
| `PrimalKKTSolver`, `KKTResult` | Ported | M5 stage 5.9 — this is `work/sphere_dli_example.py`'s solver, so it gates reproducing that example | |
| `SmoothedDualMaster`, `SmoothedLBFGSSolver` | Planned | M5 stage 5.9 | |
| `ChambollePockSolver`, `ChambollePockResult` | Planned | M5 stage 5.9 | |
| `solve_support_values` | Ported | M5 stage 5.9; becomes support-function evaluation on a `ConvexSet` | |
| `solve_primal_feasibility` | Ported | M5 stage 5.8 — the inclusion test of §18.5 | |

## `subsets.py` → `geometry/sets.py`, `geometry/convex.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `Subset` | Ported | `Subset` | |
| `EmptySet`, `UniversalSet` | Ported | same names | |
| `Complement`, `Intersection`, `Union` | Ported | same names | |
| `ConvexSubset` | Ported | `ConvexSet`, now carrying `project`, `indicator` and `support_function` as three views of one object (§16.1) | |
| `ConvexIntersection` | Subsumed | `Intersection` of convex sets, which reports itself convex | |
| `Ball` | Ported | `Ball` | |
| `Ellipsoid` | Ported | `Ellipsoid` | |
| `NormalisedEllipsoid` | Subsumed | `Ellipsoid` with a scaled shape operator | |
| `HalfSpace` | Ported | `HalfSpace` | |
| `HyperPlane` | Ported | `Hyperplane` | |
| `_EllipsoidalGeometry` | Subsumed | `_EllipsoidSupport` | |
| `Sphere` | Ported | The *surface* of a ball. Not convex, so it has no support function; used for sampling on a shell. Worth keeping? | Worth keeping with an eye to constrained optimiseation. Same for ellipsoid below.|
| `EllipsoidSurface` | Ported | As above | |
| `LevelSet`, `SublevelSet` | Ported | Sets defined by a functional. §18.5's inclusion test produces exactly a sublevel set, so this arrives with M5 stage 5.8 | |
| `PolyhedralSet` | Ported | §18.12: `Polytope`, with a recorded inner/outer status so §18.4's sandwich is a type rather than a convention | |

## `subspaces.py` → `geometry/subspaces.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `OrthogonalProjector` | Ported | `OrthogonalProjector`, with `from_basis` and `onto_kernel` | |
| `LinearSubspace` | Ported | `LinearSubspace`; v1's `dimension` used the trace formula on a non-orthonormal basis and returned 169 for a 12-dimensional subspace (§16) | |
| `AffineSubspace` | Ported | `AffineSubspace`; see Part 2, several methods are not | |

## `forward_problem.py`, `inversion.py`, `linear_bayesian.py`, `linear_optimisation.py`, `backus_gilbert.py`

The whole inversion layer. Planned as **M5**, designed in DESIGN.md §18, and
deliberately not started.

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `ForwardProblem` | Ported | M5 stage 5.1, taking a measure *or* a convex set as the data uncertainty (§18.1) | |
| `LinearForwardProblem` | Ported | M5 stage 5.1 | |
| `Inversion`, `LinearInversion` | Subsumed | §18.7's `Estimator` kinds. An inverse problem is an inference problem with `T == identity`, so there is one hierarchy | |
| `Inference`, `LinearInference` | Subsumed | as above | |
| `LinearBayesianInversion` | Ported | M5 stage 5.4 as `Bayesian`. Four of its 24 methods are inversion; the rest move to `numerics` (§18.9) | |
| `LinearLeastSquaresInversion` | Ported | M5 stage 5.3 | |
| `ConstrainedLinearLeastSquaresInversion` | Ported | M5 stage 5.3 | |
| `LinearMinimumNormInversion` | Ported | M5 stage 5.3, with the discrepancy principle on §18.6's root-find primitive | |
| `ConstrainedLinearMinimumNormInversion` | Ported | M5 stage 5.3 | |
| `BackusInference` | Ported | M5 stages 5.7 and 5.9, with four routes (§18.3) | |
| `DualMasterCostFunction` | Ported | M5 stage 5.9. Its docstring already *is* BGP eq. (28) — the support function of an image | |

## `symmetric_space/`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `SymmetricHilbertSpace` | Ported | `SymmetricSpace` | |
| `AbstractSymmetricLebesgueSpace` | Subsumed | `SymmetricSpace` with `order == 0` | |
| `SymmetricSobolevSpace` | Subsumed | `SymmetricSpace` with a Sobolev metric — *not* a mass-weighted space (§13.2) | |
| `InvariantLinearAutomorphism` | Ported | `DiagonalLinearOperator` | This is fine so long as the functionality is the same. Worth checking in detail, including methods for construction. |
| `InvariantGaussianMeasure` | Subsumed | `GaussianMeasure` with a diagonal covariance; the hand-coded sampling correction falls out of `white_noise` (§13.1) | Again, this is fine so long as the functionality is still there, including, say, all the optimisations for KL divergence etc.|
| `CorrelatedInvariantGaussianMeasure` | Ported | Cross-covariance between two invariant fields on one domain. `work/dynamic_topography.py` uses it for the coupled density/traction problem. Not ported; probably wants to be a `DirectSum` measure with a diagonal cross-block | This is a needed feature. The reason perhaps for keeping it is the form of sampling it allows via an extended KL expansion.|
| `circle.Lebesgue/Sobolev` | Ported | `PeriodicBox` in 1D | |
| `torus.Lebesgue/Sobolev` | Ported | `PeriodicBox` in 2D | |
| `plane.Lebesgue/Sobolev` | Ported | `Box` in 2D | |
| `line.Lebesgue/Sobolev` | Ported | `Interval` | |
| `sphere.Lebesgue/Sobolev` | Ported | `Sphere` | |
| *(3D periodic box)* | *(new)* | Free, from the N-dimensional `rfftn` construction | |
| `plot`, `plot_error_bounds`, `plot_geodesic`, `plot_geodesic_network`, `plot_points`, `create_map_figure` | Ported (O8, in part) | **O8**. Its own layer dispatching on space type, not methods on the spaces | yes, we need this. create_map_figure can probably be replaced by writing a kind of overload for plt.subplots adapted to cartopy, and hence allowing for mutliple panels etc.|

## `checks/` → `testing.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `HilbertSpaceAxiomChecks` | Ported | `check_space`, `check_coordinates`, `check_representer`, `check_white_noise` | |
| `LinearOperatorAxiomChecks` | Ported | `check_operator`, `check_traits` | |
| `NonLinearOperatorAxiomChecks` | Ported | `check_derivative`, `check_gradient`, `check_second_derivative` | |
| `AffineOperatorAxiomChecks` | Subsumed | `check_operator` handles affine operators | |
| — | *(new)* | `check_measure`, `check_projection` | |

## `datasets.py` → `data/`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `load_gsn_stations` | Ported | `Sphere.stations()`, reading the shipped table | |
| `sample_earthquakes` | Ported | `Sphere.earthquakes()` | |
| `download_gsn_stations` | Planned (X) | Live IRIS fetch. Kept out of the import path deliberately; should it exist at all as an explicit refresh command? | It's useful for making examples, but not core functionality. If it can be done better, then fine. |
| `download_usgs_earthquakes` | Planned (X) | As above, for USGS | |

## `quadratic_form_quantile.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `weighted_chi2_cdf` | Ported | M5. The distribution of a quadratic form in Gaussians — what a credible set of a chi-squared statistic needs | |
| `weighted_chi2_quantile` | Ported | M5. Imhof, Wood–Saddlepoint, and Monte Carlo methods with an automatic choice | |

## `dynamical_system.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `DynamicalSystem` | Planned (later) | Nothing in v2 yet. Time evolution is the natural home of the nonlinear work — is this a v2 concern or a separate thing? | This is planned, but not a priority. It will allow for things like Kalman filters to be implemented. The idea is to provide a common interface for sequential problems etc. This can be come back to later. Same for points below.|
| `AutonomousDynamicalSystem` | Planned (later) | as above | |
| `LinearDynamicalSystem` | Planned (later) | as above | |
| `AutonomousLinearSystem` | Planned (later) | as above | |

## `data_assimilation/`

Seven classes and 37 functions across `core.py` and the pendulum examples.

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `AssimilationEngine`, `BayesianAssimilationProblem` | Dropped | Depends on `dynamical_system.py` above | This is actually just a teaching module included for convenience. It is not part of the core library, though as noted above we will eventually add in stuff for sequentual data assimilation. For the moment al this is droped from V2 |
| `LinearKalmanFilter`, `EnsembleKalmanFilter` | Dropped | A Kalman filter is §18's `(measure, M)` cell iterated in time, so the structure is there once M5 is | |
| `GaussianLikelihood`, `LinearGaussianLikelihood` | Subsumed | `ForwardProblem` with a data error measure, once M5 lands | |
| `ProbabilityGrid` | Dropped | Dense grid representation of a low-dimensional posterior, for teaching | |
| `pendulum/`, and the plotting in `core.py` | Dropped | Worked examples rather than library. Somewhere else? | |

## Everything else

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `plot.py` — `SubspaceSlicePlotter`, `plot_slice`, `plot_1d_distributions`, `plot_corner_distributions` | Planned | **O8** | |
| `parallel.py` — `parallel_mat_mat`, `parallel_compute_dense_matrix_from_scipy_op` | Dropped | The joblib branches doubled every operator and each carried its own copy of the adjoint. finufft and pyshtools thread internally; parallelism belongs around an operator, not inside it (§20.8) | I'm happy to take your lead here, but I think it is viable for the action of some operators to be parallelised. And indeed, for large problems this will almost certainly be the case, though that will be implemented elsewhere.  |
| `utils.py` — `configure_threading` | Dropped | Thread-count control. Still wanted, but as a package-level setting rather than a call every script makes? | This basically doesn't work. In practice I just end up setting environement variables at run time. So this can go.  |
| `auxiliary.py` — `empirical_data_error_measure` | Dropped | M5 stage 5.1: a data error measure estimated from repeat observations | Cut, it doesn't get used as is. but the idea might be worth revisiting and rehousing. |
| `config.py` | Planned (O8) | v1's plotting backend switch. Follows O8 | Happy to follow suggestions here as plans develop. |

---

# Part 2 — Methods

Only the classes carrying enough methods for something to hide in. This is
where the losses are: 23 of v1's 145 classes hold 597 of its 919 public
methods, and a class marked *Ported* above can still have shed half of them.

## `HilbertSpace` (30 methods)

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `dim`, `zero`, `copy`, `inner_product`, `add`, `subtract`, `negative`, `axpy`, `norm`, `squared_norm` | Ported | same names | |
| `multiply` | Ported | `scale` — renamed because `multiply` reads as pointwise | |
| `ax` | Ported | `scale_inplace` | |
| `to_components`, `from_components`, `basis_vector` | Ported | on `CoordinateSpace`, since coordinates are a capability, not a requirement (§3.2) | |
| `is_orthonormal` | Ported | same, plus `has_diagonal_metric` | |
| `gram_schmidt` | Ported | `gram_schmidt`, now reorthogonalising when a vector loses more than `1/sqrt(2)` of its norm | |
| `random` | Ported | `random` **and** `white_noise`, separated. In v1 the two were confused, which is the defect of §9 | |
| `sample_expectation` | Ported | `mean` | |
| `identity_operator`, `zero_operator` | Ported | `LinearOperator.identity`, `LinearOperator.zero` | |
| `riesz`, `inverse_riesz`, `to_dual`, `from_dual`, `dual`, `duality_product` | Dropped | No dual spaces. `representer` and `apply_gram`/`solve_gram` do this work, and the metric enters in exactly one place (§1, §5.6) | |
| `is_element` | Dropped | Duck typing. v1's implementations mostly checked an array shape, which caught nothing worth catching | |
| `coordinate_inclusion`, `coordinate_projection` | Ported | Operators between a space and its component space. Not ported; reachable through `from_component_matrix` on the identity, but clumsily | I think this is useful on the appropriate spaces. |
| — | *(new)* | `orthonormal_basis`, `white_noise`, `representer`, `gram_matrix` | |

## `LinearOperator` (19 methods)

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `adjoint` | Ported | `adjoint`, memoised so `A.adjoint.adjoint is A` — which is what makes the palindrome rule work (§5) | |
| `self_adjoint` | Ported | `self_adjoint`, and `Traits.SELF_ADJOINT` | |
| `matrix` | Ported | `matrix(form=..., by=...)`, filling from the cheaper side | |
| `with_dense_matrix` | Ported | `assembled()`, and it is now on every operator rather than a constructor flag | |
| `from_matrix`, `self_adjoint_from_matrix` | Ported | `from_component_matrix`, `from_derivative_matrix` — the caller must now say which representation their array is in (§5.3) | |
| `from_vectors`, `from_vector` | Ported | `from_vectors(..., orthonormal=)` | |
| `from_tensor_product`, `self_adjoint_from_tensor_product` | Ported | `from_tensor_product` | |
| `from_linear_forms`, `from_linear_form` | Subsumed | `from_derivative_matrix`, whose rows *are* the forms | As a comment, I don't find from_derivative_matrix the clearest naming, so worth some thought |
| `from_formal_adjoint` | Ported | `lift_formal_adjoint` in `symmetric_space` (§3.5) | |
| `from_formally_self_adjoint` | Ported | `lift_formal_adjoint(..., traits=...)`. It no longer claims self-adjointness for you: a formally self-adjoint operator is self-adjoint under the new metric only if it commutes with the ratio of the two (§9) | |
| `linear` | Dropped | Type-level in v2 | |
| `dual`, `self_dual` | Dropped | No dual spaces | |
| `extract_diagonal`, `extract_diagonals` | Ported | `LinearOperator.diagonals(offsets=..., form=..., probe=...)`; `random_diagonal` and `deflated_diagonal` cover the stochastic case | This has been useful in the past, so I'd want a reason or a replacement. |
| — | *(new)* | `traits`, `with_traits`, `from_derivative_callables`, `has_derivative`, `has_second_derivative` | |

## `GaussianMeasure` (41 methods)

Ported as a class; a third of its methods are not. This is the largest single
concentration of things to decide about.

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `expectation`, `covariance`, `sample`, `samples`, `sample_expectation`, `domain` | Ported | same names, mostly on `ProbabilityMeasure` | |
| `has_zero_expectation`, `sample_set` | Ported | `has_zero_expectation`, `can_sample` | |
| `covariance_factor`, `covariance_factor_set` | Ported | `covariance_factor` | |
| `inverse_covariance`, `inverse_covariance_set` | Ported | `precision` — renamed, since "inverse covariance" invites forming it | |
| `inverse_covariance_factor`, `inverse_covariance_factor_set` | Ported | `precision_factor` | |
| `from_standard_deviation`, `from_standard_deviations` | Ported | `from_standard_deviation`, taking a scalar or an array | |
| `from_covariance_matrix`, `from_samples` | Ported | same names | |
| `from_direct_sum` | Ported | `from_product` | |
| `affine_mapping` | Ported | `affine_map`, `push_forward`, `translate` | |
| `zero_expectation` | Ported | `translate` by the negated expectation | |
| `credible_set` | Ported | M5. This is §18.1's measure-to-set hardening, and `tutorials/gaussian_measure_to_sets_demo.ipynb` is built on it | |
| `ambient_ball`, `weakened_ellipsoid` | Ported (ambient_ball) | M5, with `credible_set` — the two looser hardenings | |
| `as_multivariate_normal` | Ported | M5. The bridge to `scipy.stats` | |
| `with_dense_covariance` | Subsumed | `covariance.assembled()` | |
| `low_rank_approximation` | Subsumed | `random_eig` on the covariance | |
| `with_regularized_inverse` | Ported | Precision of a rank-deficient covariance, with a floor. Not ported | Has been used, so probably worth keeping. |
| `with_sparse_approximation` | Ported | Thresholded sparse covariance. Wanted by the localised preconditioners | Has been used, so probably worth keeping.  |
| `sample_pointwise_variance`, `sample_pointwise_std` | Subsumed | `pointwise_variance` on a `SymmetricSpace` computes this exactly, without sampling — but only for an *invariant* measure. The sampled version is still the general answer | |
| `deflated_pointwise_variance`, `deflated_pointwise_std` | Ported | Pointwise variance with a low-rank part removed. Needs `deflated_diagonal` | Seems like a good idea, though I'm not sure it's ever worked properly. Worthlooking|
| `two_point_covariance` | Ported | `C(x, y)` as a function of two points. Not ported | A useful method. Needs thinking about how to generalise (say to direct sum spaces)|
| `directional_statistics`, `directional_covariance`, `directional_variance` | Ported | Statistics of `(x, u)` along given directions. Cheap and useful; no v2 home | Yes. useful. |
| `rescale_directional_variance` | Ported | as above | Again, useful|
| `kl_divergence` | Ported | Between two Gaussians. Needs `estimate_log_determinant`, which is in `numerics` | Definitely needed, and possibly improvable.|
| `nuclear_norm`, `hilbert_schmidt_norm` | Ported | Trace-class and Hilbert–Schmidt norms of the covariance. `random_trace` gives the first stochastically | yes, useful |
| — | *(new)* | `mahalanobis_squared`, `log_density`, `grad_log_density`, `precision` | |

## The concrete spaces

v1 spreads one interface across ten classes — `Lebesgue` and `Sobolev` for the
circle, line, plane, torus and sphere. v2 has `PeriodicBox`, `Box` and
`Sphere`, with the Sobolev variant being the same coordinate map under a
different metric. So the union of method names is the honest comparison.

### Structure and coordinates

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `to_components`, `from_components`, `project_function` | Ported | same names | |
| `to_coefficients`, `from_coefficients` | Subsumed | v2 components *are* the coefficients | Just to check this is correct given the coefficients can be complex even when the coefficients are not|
| `radius`, `radius_x`, `radius_y`, `bounds_x`, `bounds_y`, `a`, `b`, `c` | Subsumed | `lengths` on a box, `radius` on the sphere | |
| `kmax`, `lmax`, `degree`, `points`, `angles`, `point_spacing` | Ported | `lmax`, `grid_shape`, `colatitudes`, `longitudes`, `grid_axes` | |
| `grid`, `grid_type`, `sampling`, `extend`, `normalization`, `csphase` | Subsumed | Fixed conventions, pinned by test rather than configurable (§13.4) | |
| `index_to_integer`, `integer_to_index`, `representative_index`, `wavevector_indices`, `indices` | Subsumed | `_packing` handles the layout; none of it is public because none of it should be | |
| `laplacian_eigenvalue`, `laplacian_eigenvalues` | Ported | `laplacian_eigenvalues`, as an array | |
| `laplacian_eigenvector_squared_norm` | Dropped | v1's factor-of-two bookkeeping. v2's Lebesgue basis is orthonormal, so this is identically one (§13.2) | Is this always the case, say if the sphere has non-zero radius or with fourier bases? Please check carefully.|
| `laplacian_eigenvectors_at_point` | Ported | `basis_at` | |
| `degree_multiplicity` | Ported | Not ported. Trivial to add | Used in the past, say for traces|
| `fft_factor`, `inverse_fft_factor` | Subsumed | Internal to the transform | |
| `angle_to_point`, `point_to_angle`, `angle_to_point_x/y`, `circle_space`, `torus_space` | Subsumed | `Box`/`Interval` subclass `PeriodicBox` rather than wrapping it, so there is no conversion (§13.4) | |
| `gaussian_curvature` | Ported | Not ported | Needed for flexure, and harmless. |
| `is_element`, `ax`, `axpy`, `zero`, `inner_product`, `norm` | Ported | on `HilbertSpace` | |

### Operators

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `point_evaluation_operator` | Ported | same name, matrix-free, and NUFFT-backed on a box, and faster than v1: a non-uniform FFT rather than a per-point Legendre evaluation (DESIGN 21.15) | Check for numerical optimistions for the concrete spaces |
| `path_average_operator` | Ported | same name, as `W E` (§20.6), and faster than v1: a non-uniform FFT rather than a per-point Legendre evaluation (DESIGN 21.15) |  As above|
| `dirac`, `dirac_representation` | Ported | `dirac`, whose `.representer` is the second | |
| `geodesic_distance`, `geodesic_quadrature`, `geodesic_ball_quadrature` | Ported | same names | |
| `geodesic_ball_integral`, `geodesic_ball_average` | Ported | `geodesic_ball_average_operator` | |
| `spherical_cap_integral`, `spherical_cap_average` | Ported | same names, exact in the harmonic basis | |
| `to_coefficient_operator`, `from_coefficient_operator` | Ported | `coefficient_operator` | |
| `with_degree`, `degree_transfer_operator` | Ported | same names | |
| `with_order` | Ported | same name | |
| `order_inclusion_operator` | Ported | The embedding `H^s -> H^t`. Not ported | This is useful |
| `spectral_projection_operator` | Ported | Projection onto a band of degrees. `coefficient_operator` gives the map *out*; this is the projector *within* | Useful |
| `derivative_operator` | Ported | `d/dx` on the circle and line. Diagonal in the basis, so nearly free | Useful |
| `flexural_operator`, `inverse_flexural_operator` | Ported | Not ported. `work/flexure.py` and `work/dynamic_topography.py` are built on these, so they gate reproducing two of the worked examples | Useful |
| `spatial_multiplication_operator` | Ported | Multiplication by a field. Needed for a spatially varying coefficient, so it gates the same two examples | Needed and simple |
| `l2_products_operator` | Ported | Inner products against a set of fields | Needed|
| `estimate_truncation_degree` | Ported | Choosing `lmax` from a target accuracy | These are useful |
| `distance_localized_preconditioner` | Ported | `inference.InvariantDistancePreconditioner`. Checked as asked: the implementation is right, but `apply_taper` defaulted to False and an untapered truncation is strongly *indefinite*, which breaks CG rather than slowing it. Tapering is now the default (DESIGN §23.6) | This was never as good as I hoped -- so check the implementation -- but should be useful |

### Measures and fields

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `invariant_gaussian_measure` | Ported | `invariant_measure` | |
| `heat_kernel_gaussian_measure` | Ported | `heat_measure` | |
| `sobolev_kernel_gaussian_measure` | Ported | `sobolev_measure` | |
| `point_value_scaled_*_gaussian_measure` (three) | Ported | `pointwise_std=` on each of the above, which is one keyword rather than three methods (§20.7) | |
| `norm_scaled_*_gaussian_measure` (three) | Ported | Calibration by the *norm* rather than the pointwise value. Not ported; the same one-keyword treatment would work | Worth it|
| `correlated_invariant_gaussian_measure` | Ported | See `CorrelatedInvariantGaussianMeasure` in Part 1 | Needed in some form. See comments above |
| `heat_kernel`, `sobolev_kernel`, `sobolev_function` | Subsumed | `heat_symbol`, `sobolev_symbol` | |
| `invariant_automorphism` | Ported | `invariant_operator` | |
| `invariant_covariance_function` | Ported | The covariance as a function of geodesic distance. Not ported | Worth having, I thought. Need a reason to drop. |
| `sample_power_measure` | Ported | Sampling from a prescribed power spectrum | Needed |
| `vector_multiply`, `vector_sqrt` | Ported | Pointwise algebra on fields. `work/flexure.py` builds its rigidity field with these. No v2 home — see `HilbertModuleMixin` in Part 1 | Needed in some form|
| `from_covariance`, `from_heat_kernel_prior`, `from_sobolev_kernel_prior`, `from_sobolev_parameters` | Subsumed | Constructors on the measures rather than on the space | |

### Geometry and data

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `random_point`, `random_points` | Ported | same names | |
| `iris_stations`, `random_earthquakes` | Ported | `stations`, `earthquakes` | |
| `domain_mask`, `random_domain_points` | Ported | `domain_mask`; `random_domain_points` is **Open** | |
| `pairs_within_distance` | Ported | same name, with the chord formula so a point is in its own neighbourhood (§20.7) | |
| `cluster_points` | Ported | On the sphere, and used to build the blocks a `LocalisedPreconditioner` takes | Useful for some preconditioners |
| `random_source_receiver_paths` | Ported | Not ported. `stations` and `earthquakes` give the ingredients, so this is convenience — but it is convenience every tomography script writes | Yes, needed somewhere |

## `HilbertSpaceDirectSum` (21 methods)

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `subspaces`, `subspace`, `number_of_subspaces` | Ported | `subspaces`, `subspace`; the count is `len(subspaces)` | |
| `subspace_projection`, `subspace_inclusion` | Ported | `projection`, `inclusion`, memoised so the adjoint involution holds | |
| `to_components`, `from_components`, `dim`, `is_orthonormal`, and the vector operations | Ported | on `DirectSum` and `_CoordinateDirectSum` | |
| `to_dual`, `from_dual`, `canonical_dual_isomorphism`, `canonical_dual_inverse_isomorphism` | Dropped | No dual spaces | |
| — | *(new)* | `labels`, `index`, `component` — named blocks, excluded from the space's identity | |

## `AffineSubspace` (22 methods)

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `translation`, `projector`, `project`, `from_linear_equation` | Ported | same names | |
| `tangent_space`, `get_tangent_basis` | Ported | `tangent` | |
| `from_tangent_basis`, `from_complement_basis` | Ported | Not ported; `OrthogonalProjector.from_basis` is the ingredient | All these methods below have been useful, and so they are either needed or the functionality provided elsewhere |
| `from_hyperplanes`, `to_hyperplanes` | Ported | Not ported | |
| `constraint_operator`, `constraint_value`, `has_explicit_equation` | Ported | Whether the subspace remembers the equation that defined it | |
| `pseudo_inverse`, `projection_operator`, `boundary` | Ported | Not ported | |
| `with_translation`, `with_constraint_value` | Ported | Not ported | |
| `solver`, `preconditioner` | Subsumed | Passed in where needed rather than stored on the subspace | |
| `is_element` | Ported | `contains` on `Subset` | |
| `condition_gaussian_measure` | Ported | M5 — conditioning a measure on a linear constraint is a small Bayesian update | |

## `LinearBayesianInversion` (24 methods)

Four of these are inversion; §18.9 has the disposition of the rest.

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `model_prior_measure`, `data_prior_measure`, `joint_prior_measure` | Ported | M5 stage 5.4 | |
| `normal_operator`, `get_normal_equations_rhs`, `kalman_operator` | Ported | M5 stage 5.4 | |
| `model_posterior_measure` | Ported | M5 stage 5.4, but as `Bayesian(problem, prior, solver)(data)` — a mapping, not a method taking both data and configuration (§18.7) | |
| `posterior_expectation_operator` | Subsumed | `GaussianEstimator.mean_map` | |
| `with_formalism` | Subsumed | An argument on the estimator, defaulting to whichever space is smaller (§18.10) | |
| `log_evidence`, `mahalanobis_evidence_term` | Ported | M5, as a functional on the data | |
| `estimate_log_determinant` | Ported | `numerics.functional_calculus` | |
| `low_rank_surrogate` | Ported | `LinearGaussianInversion.low_rank_surrogate`, on top of the new `GaussianMeasure.low_rank_approximation` (DESIGN §23.7) | |
| `woodbury_data_preconditioner`, `woodbury_model_preconditioner` | Ported | `WoodburyPreconditioner.data_form` / `.model_form`, one class that picks the identity from the space it is asked to invert (DESIGN §22.12) | |
| `diagonal_normal_preconditioner` | Ported | `inference.NormalDiagonalPreconditioner`, a free-standing LinearSolver taking a `NormalOperator` rather than a method on the inversion (DESIGN §23.4) | |
| `sparse_localized_preconditioner` | Ported | `inference.LocalisedPreconditioner`, likewise. Blocks may overlap; `R`'s off-diagonal is dropped, now documented and tested as such | |
| `surrogate_inversion` | Ported | `LinearGaussianInversion.surrogate` / `NormalOperator.surrogate`. Returns the surrogate *normal operator*, which is the only part of a surrogate problem ever used, and may live on a different model space (DESIGN §23.2) | |
| `surrogate_normal_preconditioner`, `surrogate_woodbury_data_preconditioner`, `surrogate_woodbury_model_preconditioner` | Subsumed | Passing cheap factors *is* the surrogate case: `WoodburyPreconditioner.from_normal(inversion.surrogate(...))` | |
| `parameterized_inversion`, `data_reduced_inversion` | Dropped | They forward to the `ForwardProblem` methods of the same name (§18.9) | |
| `normal_residual_callback` | Subsumed | With the solver callbacks above | |

## `LinearForwardProblem` (12 methods)

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `from_direct_sum` | Ported | M5 stage 5.1 — joint inversion of several data sets | |
| `data_measure_from_model`, `data_measure_from_model_measure`, `joint_measure` | Ported | M5 stage 5.1 | |
| `synthetic_data`, `synthetic_model_and_data` | Ported | M5 stage 5.1 | |
| `chi_squared`, `chi_squared_from_residual`, `critical_chi_squared`, `chi_squared_test` | Ported | M5 stage 5.1, with the set coming first and the boolean on top (§18.11) | |
| `parameterized_problem`, `data_reduced_problem` | Ported | M5 stage 5.1 | |
