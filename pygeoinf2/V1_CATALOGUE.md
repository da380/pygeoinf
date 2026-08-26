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
| **Planned** | not yet; the stage that brings it is named |
| **Dropped** | deliberately not coming, with the reason |
| **Open** | I have no recommendation. Your call. |

**The default is that everything comes across.** *Dropped* needs a reason in
the row; *Open* means I could not supply one.

The last column is for you. Overrule anything — a **Dropped** you want kept, a
**Planned** that should be sooner or later than I have it, an **Open** you can
settle. I will fold your answers back into DESIGN.md's stage lists.

## Summary

320 rows, some grouping several closely related names.

| status | rows | |
|---|---|---|
| **Ported** | 126 | in v2 now |
| **Subsumed** | 33 | reached another way |
| **Planned** | 51 | mostly M5 and O8 |
| **Dropped** | 15 | each with a reason in its row |
| **Open** | 53 | **your call** |

Part 1 is by module, at class and free-function level. Part 2 is by method,
for the classes big enough for something to hide in — 23 of v1's 145 classes
hold 597 of its 919 public methods, and that is where the losses are. A class
marked *Ported* can still have shed half of what it did.

## Where I would look first

Fifty-three Opens is a lot to read. These are the ones I think matter most, and
the first three block work that already exists:

- **`flexural_operator`, `inverse_flexural_operator`, `spatial_multiplication_operator`, `vector_multiply`, `vector_sqrt`.** `work/flexure.py` and `work/dynamic_topography.py` are built on these, so two of the four worked examples cannot be reproduced on v2 without them. `vector_multiply` in particular has no v2 home at all — it is the module structure of `HilbertModuleMixin`, and v2 dropped it without deciding to.
- **`CorrelatedInvariantGaussianMeasure`.** The coupled two-field prior. Same two examples.
- **`GaussianMeasure`'s statistics** — `directional_*`, `two_point_covariance`, `kl_divergence`, `nuclear_norm`, `hilbert_schmidt_norm`. Eleven Opens on one class. Cheap to port, and I do not know which you use.
- **GMRES.** v2 has no coordinate-free solver for a non-symmetric operator except BiCGStab.
- **`dynamical_system.py` and `data_assimilation/`.** Eleven classes with no v2 position at all, because I do not know whether they are v2's concern.

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
| `HilbertModuleMixin` | Open | Pointwise multiplication of fields. `work/flexure.py` needs it for a spatially varying rigidity. No v2 home yet — see Part 2 | |

## `linear_operators.py` → `algebra/operators.py`, `nodes.py`, `diagonal.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LinearOperator` | Ported | `LinearOperator`, with traits and memoised adjoints | |
| `MatrixLinearOperator` | Subsumed | `from_component_matrix` / `from_derivative_matrix` | |
| `DenseMatrixLinearOperator` | Subsumed | as above, plus `assembled()` | |
| `SparseMatrixLinearOperator` | Open | No sparse-backed operator in v2. `_weight_operator` in `symmetric_space/base.py` does it privately for one case; it probably wants to be public | |
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
| `FCGSolver` | Open | Flexible CG, for a preconditioner that changes between iterations. Not ported. Needed if the localised preconditioners come back | |
| `ScipyIterativeSolver` | Dropped | The point of v2's Krylov methods is that they run coordinate-free against PETSc or MFEM. A SciPy wrapper cannot | |
| `CGMatrixSolver`, `BICGMatrixSolver`, `BICGStabMatrixSolver`, `GMRESMatrixSolver` | Dropped | Matrix-only convenience wrappers around SciPy, superseded as above. **GMRES itself is missing** and is the one gap — see Open below | |
| *(GMRES, coordinate-free)* | Open | v2 has no solver for a non-symmetric operator other than BiCGStab. Worth adding? | |
| `ProgressCallback`, `ResidualTrackingCallback`, `SolutionTrackingCallback` | Planned | v2 returns a `SolveResult` with the residual history instead. Live callbacks are still wanted for long solves — no stage yet | |

## `preconditioners.py` → `numerics/preconditioners.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `IdentityPreconditioningMethod` | Ported | `IdentityPreconditioner` | |
| `JacobiPreconditioningMethod` | Ported | `JacobiPreconditioner` | |
| `BandedPreconditioningMethod` | Planned | M5. Needs `extract_diagonals` | |
| `ColumnThresholdedPreconditioningMethod` | Planned | M5 | |
| `ExactBlockPreconditioningMethod` | Planned | M5 | |
| `SpectralPreconditioningMethod` | Planned | M5; `random_eig` is already there | |
| `IterativePreconditioningMethod` | Planned | M5; wants `FCGSolver` | |

## `functional_calculus.py` → `numerics/functional_calculus.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LanczosOperatorFunction` | Ported | `OperatorFunction` | |
| `lanczos_tridiagonalize` | Ported | `lanczos_tridiagonalise` | |
| `iter_lanczos_tridiagonalize` | Ported | `iter_lanczos_tridiagonalise` | |
| `apply_operator_function` | Ported | same name | |
| `operator_function_quadratic_form` | Ported | `operator_quadratic_form` | |
| — | *(new)* | `operator_sqrt`, `operator_log`, `operator_exp`, `operator_power`, `operator_inverse_sqrt`, and dispatch to the exact route for a diagonal operator | |

## `low_rank.py` → `numerics/randomised.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `LowRankEig`, `LowRankSVD`, `LowRankCholesky` | Ported | same names | |
| `random_range` | Ported | same name; no longer routes mass-weighted spaces through the broken white-noise measure (§9) | |
| `random_trace` | Ported | same name, returning an `Estimate` with a standard error | |
| `random_diagonal` | Ported | same name (Bekas–Kokiopoulou–Saad) | |
| `deflated_diagonal` | Open | Not ported. Diagonal estimation with a low-rank part removed first | |
| `white_noise_measure` | Dropped | The v1 defect of §9: it produced covariance `G`, not `I`. Replaced by `HilbertSpace.white_noise` | |

## `nonlinear_optimisation.py` → `numerics/optimisation.py`, `line_search.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `ScipyUnconstrainedOptimiser` | Dropped | It passed `G^-1 dJ/dc` as `jac` while passing the Galerkin matrix as `hess` — mixing gradients and derivatives in one call (§12). Replaced by bespoke coordinate-free methods | |
| `line_search` | Ported | `ArmijoLineSearch`, `StrongWolfeLineSearch` | |
| — | *(new)* | `SteepestDescent`, `NonlinearCG`, `LBFGS`, `NewtonCG`, `TrustRegionNewton`, `truncated_cg`, `gauss_newton_hessian` | |

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
| `CallableSupportFunction` | Planned | §18.12: `ConvexSet.from_support_function`, the oracle case M5 route (d) needs | |

## `convex_optimisation.py` → `numerics/convex.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `SubgradientDescent`, `SubgradientResult` | Ported | `SubgradientDescent`, `_ConvexResult` | |
| `Cut`, `Bundle` | Planned | M5 stage 5.9 | |
| `ProximalBundleMethod`, `LevelBundleMethod`, `BundleResult` | Planned | M5 stage 5.9 — the dual route for a general convex prior | |
| `QPSolver`, `QPResult` | Planned | M5 stage 5.9 | |
| `SciPyQPSolver`, `OSQPQPSolver`, `ClarabelQPSolver` | Planned | M5 stage 5.9. Coordinates are fine here: the QP lives in a finite-dimensional, canonically Euclidean space | |
| `best_available_qp_solver` | Planned | M5 stage 5.9 | |
| `PrimalKKTSolver`, `KKTResult` | Planned | M5 stage 5.9 — this is `work/sphere_dli_example.py`'s solver, so it gates reproducing that example | |
| `SmoothedDualMaster`, `SmoothedLBFGSSolver` | Planned | M5 stage 5.9 | |
| `ChambollePockSolver`, `ChambollePockResult` | Planned | M5 stage 5.9 | |
| `solve_support_values` | Planned | M5 stage 5.9; becomes support-function evaluation on a `ConvexSet` | |
| `solve_primal_feasibility` | Planned | M5 stage 5.8 — the inclusion test of §18.5 | |

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
| `Sphere` | Open | The *surface* of a ball. Not convex, so it has no support function; used for sampling on a shell. Worth keeping? | |
| `EllipsoidSurface` | Open | As above | |
| `LevelSet`, `SublevelSet` | Planned | Sets defined by a functional. §18.5's inclusion test produces exactly a sublevel set, so this arrives with M5 stage 5.8 | |
| `PolyhedralSet` | Planned | §18.12: `Polytope`, with a recorded inner/outer status so §18.4's sandwich is a type rather than a convention | |

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
| `ForwardProblem` | Planned | M5 stage 5.1, taking a measure *or* a convex set as the data uncertainty (§18.1) | |
| `LinearForwardProblem` | Planned | M5 stage 5.1 | |
| `Inversion`, `LinearInversion` | Subsumed | §18.7's `Estimator` kinds. An inverse problem is an inference problem with `T == identity`, so there is one hierarchy | |
| `Inference`, `LinearInference` | Subsumed | as above | |
| `LinearBayesianInversion` | Planned | M5 stage 5.4 as `Bayesian`. Four of its 24 methods are inversion; the rest move to `numerics` (§18.9) | |
| `LinearLeastSquaresInversion` | Planned | M5 stage 5.3 | |
| `ConstrainedLinearLeastSquaresInversion` | Planned | M5 stage 5.3 | |
| `LinearMinimumNormInversion` | Planned | M5 stage 5.3, with the discrepancy principle on §18.6's root-find primitive | |
| `ConstrainedLinearMinimumNormInversion` | Planned | M5 stage 5.3 | |
| `BackusInference` | Planned | M5 stages 5.7 and 5.9, with four routes (§18.3) | |
| `DualMasterCostFunction` | Planned | M5 stage 5.9. Its docstring already *is* BGP eq. (28) — the support function of an image | |

## `symmetric_space/`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `SymmetricHilbertSpace` | Ported | `SymmetricSpace` | |
| `AbstractSymmetricLebesgueSpace` | Subsumed | `SymmetricSpace` with `order == 0` | |
| `SymmetricSobolevSpace` | Subsumed | `SymmetricSpace` with a Sobolev metric — *not* a mass-weighted space (§13.2) | |
| `InvariantLinearAutomorphism` | Ported | `DiagonalLinearOperator` | |
| `InvariantGaussianMeasure` | Subsumed | `GaussianMeasure` with a diagonal covariance; the hand-coded sampling correction falls out of `white_noise` (§13.1) | |
| `CorrelatedInvariantGaussianMeasure` | Open | Cross-covariance between two invariant fields on one domain. `work/dynamic_topography.py` uses it for the coupled density/traction problem. Not ported; probably wants to be a `DirectSum` measure with a diagonal cross-block | |
| `circle.Lebesgue/Sobolev` | Ported | `PeriodicBox` in 1D | |
| `torus.Lebesgue/Sobolev` | Ported | `PeriodicBox` in 2D | |
| `plane.Lebesgue/Sobolev` | Ported | `Box` in 2D | |
| `line.Lebesgue/Sobolev` | Ported | `Interval` | |
| `sphere.Lebesgue/Sobolev` | Ported | `Sphere` | |
| *(3D periodic box)* | *(new)* | Free, from the N-dimensional `rfftn` construction | |
| `plot`, `plot_error_bounds`, `plot_geodesic`, `plot_geodesic_network`, `plot_points`, `create_map_figure` | Planned | **O8**. Its own layer dispatching on space type, not methods on the spaces | |

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
| `download_gsn_stations` | Open | Live IRIS fetch. Kept out of the import path deliberately; should it exist at all as an explicit refresh command? | |
| `download_usgs_earthquakes` | Open | As above, for USGS | |

## `quadratic_form_quantile.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `weighted_chi2_cdf` | Planned | M5. The distribution of a quadratic form in Gaussians — what a credible set of a chi-squared statistic needs | |
| `weighted_chi2_quantile` | Planned | M5. Imhof, Wood–Saddlepoint, and Monte Carlo methods with an automatic choice | |

## `dynamical_system.py`

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `DynamicalSystem` | Open | Nothing in v2 yet. Time evolution is the natural home of the nonlinear work — is this a v2 concern or a separate thing? | |
| `AutonomousDynamicalSystem` | Open | as above | |
| `LinearDynamicalSystem` | Open | as above | |
| `AutonomousLinearSystem` | Open | as above | |

## `data_assimilation/`

Seven classes and 37 functions across `core.py` and the pendulum examples.

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `AssimilationEngine`, `BayesianAssimilationProblem` | Open | Depends on `dynamical_system.py` above | |
| `LinearKalmanFilter`, `EnsembleKalmanFilter` | Open | A Kalman filter is §18's `(measure, M)` cell iterated in time, so the structure is there once M5 is | |
| `GaussianLikelihood`, `LinearGaussianLikelihood` | Subsumed | `ForwardProblem` with a data error measure, once M5 lands | |
| `ProbabilityGrid` | Open | Dense grid representation of a low-dimensional posterior, for teaching | |
| `pendulum/`, and the plotting in `core.py` | Open | Worked examples rather than library. Somewhere else? | |

## Everything else

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `plot.py` — `SubspaceSlicePlotter`, `plot_slice`, `plot_1d_distributions`, `plot_corner_distributions` | Planned | **O8** | |
| `parallel.py` — `parallel_mat_mat`, `parallel_compute_dense_matrix_from_scipy_op` | Dropped | The joblib branches doubled every operator and each carried its own copy of the adjoint. finufft and pyshtools thread internally; parallelism belongs around an operator, not inside it (§20.8) | |
| `utils.py` — `configure_threading` | Open | Thread-count control. Still wanted, but as a package-level setting rather than a call every script makes? | |
| `auxiliary.py` — `empirical_data_error_measure` | Planned | M5 stage 5.1: a data error measure estimated from repeat observations | |
| `config.py` | Open | v1's plotting backend switch. Follows O8 | |

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
| `coordinate_inclusion`, `coordinate_projection` | Open | Operators between a space and its component space. Not ported; reachable through `from_component_matrix` on the identity, but clumsily | |
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
| `from_linear_forms`, `from_linear_form` | Subsumed | `from_derivative_matrix`, whose rows *are* the forms | |
| `from_formal_adjoint` | Ported | `lift_formal_adjoint` in `symmetric_space` (§3.5) | |
| `from_formally_self_adjoint` | Ported | `lift_formal_adjoint(..., traits=...)`. It no longer claims self-adjointness for you: a formally self-adjoint operator is self-adjoint under the new metric only if it commutes with the ratio of the two (§9) | |
| `linear` | Dropped | Type-level in v2 | |
| `dual`, `self_dual` | Dropped | No dual spaces | |
| `extract_diagonal`, `extract_diagonals` | Open | Not ported. `random_diagonal` covers the stochastic case; the exact one is what the banded preconditioner needs | |
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
| `credible_set` | Planned | M5. This is §18.1's measure-to-set hardening, and `tutorials/gaussian_measure_to_sets_demo.ipynb` is built on it | |
| `ambient_ball`, `weakened_ellipsoid` | Planned | M5, with `credible_set` — the two looser hardenings | |
| `as_multivariate_normal` | Planned | M5. The bridge to `scipy.stats` | |
| `with_dense_covariance` | Subsumed | `covariance.assembled()` | |
| `low_rank_approximation` | Subsumed | `random_eig` on the covariance | |
| `with_regularized_inverse` | Open | Precision of a rank-deficient covariance, with a floor. Not ported | |
| `with_sparse_approximation` | Open | Thresholded sparse covariance. Wanted by the localised preconditioners | |
| `sample_pointwise_variance`, `sample_pointwise_std` | Subsumed | `pointwise_variance` on a `SymmetricSpace` computes this exactly, without sampling — but only for an *invariant* measure. The sampled version is still the general answer | |
| `deflated_pointwise_variance`, `deflated_pointwise_std` | Open | Pointwise variance with a low-rank part removed. Needs `deflated_diagonal` | |
| `two_point_covariance` | Open | `C(x, y)` as a function of two points. Not ported | |
| `directional_statistics`, `directional_covariance`, `directional_variance` | Open | Statistics of `(x, u)` along given directions. Cheap and useful; no v2 home | |
| `rescale_directional_variance` | Open | as above | |
| `kl_divergence` | Open | Between two Gaussians. Needs `estimate_log_determinant`, which is in `numerics` | |
| `nuclear_norm`, `hilbert_schmidt_norm` | Open | Trace-class and Hilbert–Schmidt norms of the covariance. `random_trace` gives the first stochastically | |
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
| `to_coefficients`, `from_coefficients` | Subsumed | v2 components *are* the coefficients | |
| `radius`, `radius_x`, `radius_y`, `bounds_x`, `bounds_y`, `a`, `b`, `c` | Subsumed | `lengths` on a box, `radius` on the sphere | |
| `kmax`, `lmax`, `degree`, `points`, `angles`, `point_spacing` | Ported | `lmax`, `grid_shape`, `colatitudes`, `longitudes`, `grid_axes` | |
| `grid`, `grid_type`, `sampling`, `extend`, `normalization`, `csphase` | Subsumed | Fixed conventions, pinned by test rather than configurable (§13.4) | |
| `index_to_integer`, `integer_to_index`, `representative_index`, `wavevector_indices`, `indices` | Subsumed | `_packing` handles the layout; none of it is public because none of it should be | |
| `laplacian_eigenvalue`, `laplacian_eigenvalues` | Ported | `laplacian_eigenvalues`, as an array | |
| `laplacian_eigenvector_squared_norm` | Dropped | v1's factor-of-two bookkeeping. v2's Lebesgue basis is orthonormal, so this is identically one (§13.2) | |
| `laplacian_eigenvectors_at_point` | Ported | `basis_at` | |
| `degree_multiplicity` | Open | Not ported. Trivial to add | |
| `fft_factor`, `inverse_fft_factor` | Subsumed | Internal to the transform | |
| `angle_to_point`, `point_to_angle`, `angle_to_point_x/y`, `circle_space`, `torus_space` | Subsumed | `Box`/`Interval` subclass `PeriodicBox` rather than wrapping it, so there is no conversion (§13.4) | |
| `gaussian_curvature` | Open | Not ported | |
| `is_element`, `ax`, `axpy`, `zero`, `inner_product`, `norm` | Ported | on `HilbertSpace` | |

### Operators

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `point_evaluation_operator` | Ported | same name, matrix-free, and NUFFT-backed on a box | |
| `path_average_operator` | Ported | same name, as `W E` (§20.6) | |
| `dirac`, `dirac_representation` | Ported | `dirac`, whose `.representer` is the second | |
| `geodesic_distance`, `geodesic_quadrature`, `geodesic_ball_quadrature` | Ported | same names | |
| `geodesic_ball_integral`, `geodesic_ball_average` | Ported | `geodesic_ball_average_operator` | |
| `spherical_cap_integral`, `spherical_cap_average` | Ported | same names, exact in the harmonic basis | |
| `to_coefficient_operator`, `from_coefficient_operator` | Ported | `coefficient_operator` | |
| `with_degree`, `degree_transfer_operator` | Ported | same names | |
| `with_order` | Ported | same name | |
| `order_inclusion_operator` | Open | The embedding `H^s -> H^t`. Not ported | |
| `spectral_projection_operator` | Open | Projection onto a band of degrees. `coefficient_operator` gives the map *out*; this is the projector *within* | |
| `derivative_operator` | Open | `d/dx` on the circle and line. Diagonal in the basis, so nearly free | |
| `flexural_operator`, `inverse_flexural_operator` | **Open** | Not ported. `work/flexure.py` and `work/dynamic_topography.py` are built on these, so they gate reproducing two of the worked examples | |
| `spatial_multiplication_operator` | **Open** | Multiplication by a field. Needed for a spatially varying coefficient, so it gates the same two examples | |
| `l2_products_operator` | Open | Inner products against a set of fields | |
| `estimate_truncation_degree` | Open | Choosing `lmax` from a target accuracy | |
| `distance_localized_preconditioner` | Planned | M5, with the other preconditioners; needs `pairs_within_distance`, which is ported | |

### Measures and fields

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `invariant_gaussian_measure` | Ported | `invariant_measure` | |
| `heat_kernel_gaussian_measure` | Ported | `heat_measure` | |
| `sobolev_kernel_gaussian_measure` | Ported | `sobolev_measure` | |
| `point_value_scaled_*_gaussian_measure` (three) | Ported | `pointwise_std=` on each of the above, which is one keyword rather than three methods (§20.7) | |
| `norm_scaled_*_gaussian_measure` (three) | Open | Calibration by the *norm* rather than the pointwise value. Not ported; the same one-keyword treatment would work | |
| `correlated_invariant_gaussian_measure` | Open | See `CorrelatedInvariantGaussianMeasure` in Part 1 | |
| `heat_kernel`, `sobolev_kernel`, `sobolev_function` | Subsumed | `heat_symbol`, `sobolev_symbol` | |
| `invariant_automorphism` | Ported | `invariant_operator` | |
| `invariant_covariance_function` | Open | The covariance as a function of geodesic distance. Not ported | |
| `sample_power_measure` | Open | Sampling from a prescribed power spectrum | |
| `vector_multiply`, `vector_sqrt` | **Open** | Pointwise algebra on fields. `work/flexure.py` builds its rigidity field with these. No v2 home — see `HilbertModuleMixin` in Part 1 | |
| `from_covariance`, `from_heat_kernel_prior`, `from_sobolev_kernel_prior`, `from_sobolev_parameters` | Subsumed | Constructors on the measures rather than on the space | |

### Geometry and data

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `random_point`, `random_points` | Ported | same names | |
| `iris_stations`, `random_earthquakes` | Ported | `stations`, `earthquakes` | |
| `domain_mask`, `random_domain_points` | Ported | `domain_mask`; `random_domain_points` is **Open** | |
| `pairs_within_distance` | Ported | same name, with the chord formula so a point is in its own neighbourhood (§20.7) | |
| `cluster_points` | Open | Not ported | |
| `random_source_receiver_paths` | Open | Not ported. `stations` and `earthquakes` give the ingredients, so this is convenience — but it is convenience every tomography script writes | |

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
| `from_tangent_basis`, `from_complement_basis` | Open | Not ported; `OrthogonalProjector.from_basis` is the ingredient | |
| `from_hyperplanes`, `to_hyperplanes` | Open | Not ported | |
| `constraint_operator`, `constraint_value`, `has_explicit_equation` | Open | Whether the subspace remembers the equation that defined it | |
| `pseudo_inverse`, `projection_operator`, `boundary` | Open | Not ported | |
| `with_translation`, `with_constraint_value` | Open | Not ported | |
| `solver`, `preconditioner` | Subsumed | Passed in where needed rather than stored on the subspace | |
| `is_element` | Ported | `contains` on `Subset` | |
| `condition_gaussian_measure` | Planned | M5 — conditioning a measure on a linear constraint is a small Bayesian update | |

## `LinearBayesianInversion` (24 methods)

Four of these are inversion; §18.9 has the disposition of the rest.

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `model_prior_measure`, `data_prior_measure`, `joint_prior_measure` | Planned | M5 stage 5.4 | |
| `normal_operator`, `get_normal_equations_rhs`, `kalman_operator` | Planned | M5 stage 5.4 | |
| `model_posterior_measure` | Planned | M5 stage 5.4, but as `Bayesian(problem, prior, solver)(data)` — a mapping, not a method taking both data and configuration (§18.7) | |
| `posterior_expectation_operator` | Subsumed | `GaussianEstimator.mean_map` | |
| `with_formalism` | Subsumed | An argument on the estimator, defaulting to whichever space is smaller (§18.10) | |
| `log_evidence`, `mahalanobis_evidence_term` | Planned | M5, as a functional on the data | |
| `estimate_log_determinant` | Ported | `numerics.functional_calculus` | |
| `low_rank_surrogate` | Ported | `numerics.randomised` | |
| `diagonal_normal_preconditioner`, `sparse_localized_preconditioner`, `woodbury_data_preconditioner`, `woodbury_model_preconditioner` | Planned | `numerics.preconditioners`, M5 | |
| `surrogate_inversion`, `surrogate_normal_preconditioner`, `surrogate_woodbury_data_preconditioner`, `surrogate_woodbury_model_preconditioner` | Subsumed | A surrogate is a transformed *problem*; the preconditioner then follows from it, so these four collapse into the four above | |
| `parameterized_inversion`, `data_reduced_inversion` | Dropped | They forward to the `ForwardProblem` methods of the same name (§18.9) | |
| `normal_residual_callback` | Planned | With the solver callbacks above | |

## `LinearForwardProblem` (12 methods)

| v1 | Status | v2 / reason | Your notes |
|---|---|---|---|
| `from_direct_sum` | Planned | M5 stage 5.1 — joint inversion of several data sets | |
| `data_measure_from_model`, `data_measure_from_model_measure`, `joint_measure` | Planned | M5 stage 5.1 | |
| `synthetic_data`, `synthetic_model_and_data` | Planned | M5 stage 5.1 | |
| `chi_squared`, `chi_squared_from_residual`, `critical_chi_squared`, `chi_squared_test` | Planned | M5 stage 5.1, with the set coming first and the boolean on top (§18.11) | |
| `parameterized_problem`, `data_reduced_problem` | Planned | M5 stage 5.1 | |
