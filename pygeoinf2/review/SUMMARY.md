# The review, consolidated

Seven layer reviews, 2026-10-29: algebra, numerics, probability, inference,
geometry, spaces, examples. Totals: 33 Must-fix (28 distinct, three found by
more than one reviewer), 96 Should-fix, 71 Could-fix. Every Must-fix below
marked ✔ was reproduced by Claude on top of the reviewer's run; the rest
were reproduced by the reviewer with the code in its file.

## Must-fix, grouped into work packages

**A. Measures that give wrong answers** (`probability.md`)
1. ✔ Derived samplers draw about twice the mean: a measure that samples
   through a callable (every posterior, conditioned measure, most sums)
   after `translate`, a push-forward or a sum. `expectation` is right, so
   only a sampling check sees it. Three sites in `gaussian.py`.
2. ✔ The correlated product's precision factor assumes the factor's
   precision factor is its exact inverse, which the contract does not
   promise; wrong precision after a sign flip.
3. `from_covariance_matrix` symmetrizes before checking, so the wrong
   `form=` becomes a different covariance instead of a refusal.
4. The identity covariance on a function space passes seven routes
   (`from_standard_deviations`, the constructor, `from_covariance_matrix`,
   `invariant_measure`, `heat_measure(0)`, `spectral_measure`,
   `correlated_measure`, a flat `power_measure`); tutorials 9 and 13 and
   `test_laplace` build one. Found by probability, spaces and examples.

**B. Absolute thresholds in numerics** (`numerics.md`), one disease
5. ✔ `weighted_chi2_quantile` depends on the units of its weights
   (`np.allclose` with an absolute tolerance, Imhof's floors): the 95% ball
   covers 0.2% of draws when the deviations are 3e-5.
6. The Lanczos calculus preallocates a `dim x dim` buffer: 107 GiB at
   dimension 120 000 for an operator with five eigenvalues.
7. Lanczos's breakdown test is floored at one: a 14% wrong square root at
   norm 1e-13, silently.
8. GMRES and MINRES report convergence on a singular operator whose
   Krylov space closes exactly, true residual of order one.
9. Chambolle-Pock stops on an absolute feasibility residual and calls
   feasibility convergence.
10. The proximal bundle method returns its start as converged on an
    objective scaled by 1e-4.

**C. Feasible sets** (`geometry.md`)
11. ✔ The default outer ellipsoid (data-space S-procedure) is not
    certified when the data outnumber the model: misses attained supports
    by up to 24 000 times; the model-space route is right to 1e-13; the
    docstring claims a check the code does not make.
12. ✔ A flat closed-form set (kernel smaller than the property) inverts a
    singular shape: indefinite precision, wrong `contains`, wrong `project`.
13. ✔ A zero-radius noise ball with a centre loses its centre.
14. ✔ `outside()` proves contact points outside when the support is negative.
15. ✔ The dual's emptiness answer reports a stand-in level of zero, so a
    prior with a negative level is declared empty.
16. ✔ The dual declares emptiness by an absolute threshold of 1e6 in
    property units.

**D. Estimators** (`inference.md`)
17. ✔ A nonzero noise expectation is ignored by the low-rank posterior's
    linear constructor, `ConstrainedLeastSquares` and `BackusGilbert`.
18. ✔ `LaplaceApproximation` always uses the Gauss-Newton curvature;
    `gauss_newton=` has no effect on it, contrary to its docstring.
19. ✔ An exact-data posterior carries a unit-noise density.
20. `resolution` is a property where `propagated_covariance()` is a method
    (shape only).
21. (From `examples.md`, Claude's diagnosis) The low-rank linear mean keeps
    the prior in the dropped directions, so a dropped eigenvalue above one
    gives an unshrunk update: 750 times too large at rank 4. The update
    must be kept only in the retained directions, each shrunk by
    `1 / (1 + lambda)`.

**E. Mechanics in algebra and spaces** (`algebra.md`, `spaces.md`)
22. ✔ Block operators (`BlockOperator`, `ColumnOperator`, `RowOperator`,
    `BlockDiagonalOperator` and their linear forms) cannot be unpickled:
    every `n_jobs=` loop over a product, layered or correlated measure, or
    a posterior from one, dies with `BrokenProcessPool`.
23. The ray travel-time operator has no order guard and memoizes traced
    rays by object identity, so an in-place update returns stale values.
24. A bounded `Box`, `Line` or `Plane` silently wraps points outside its
    domain where the spectral-element spaces refuse them.
25. Point evaluation's matrix-free switch is `dense=` on symmetric spaces
    and `form=` on spectral-element ones, and `form=` means components
    versus Galerkin everywhere else.
26. The top-level `Interval` factory (a Fourier box) collides with the
    class `interval.Interval`.
27. `ShellTransform` is exported publicly though scheduled for deletion.
28. `parallel_map(threads=)` is public with no caller.
    (`AffineOperator` treating a callable translation as lazy: Must only if
    callable vectors are in the backend contract; no shipped backend has
    them. Proposed: Should.)

**F. Examples and tutorials** (`examples.md`)
29. Tutorials 9 and 13 teach the identity covariance (closed by 4).
30. No example shows a point answer; least squares, minimum norm, the
    Backus-Gilbert error budget, the low-rank posterior, the correlated
    product, `ambient_ball(operator=)` and the parameterized problem the
    apps use appear in no example or tutorial; 51 of 122 exports appear
    nowhere. Four new examples and a gap-free renumbering proposed.
31. Three scripts say the opposite of what they print (tutorial 7's bias,
    tutorial 13's L-BFGS agreement, example 22's CG NaN).

## Should-fix items that argue against a settled decision

- D-131's `unsafe=True` bypass in `chi_squared`: on a data space that
  discretizes a function space the potential grows with resolution
  (6.8, 31, 133 at 16, 64, 256) while `consistency_set` refuses the same.
- The model-space evidence volume diverges under refinement; its
  stochastic estimate has standard error ±46 against ±0.8 in the data space.
- D-47's Woodbury inner CG at `rtol=1e-3` as a default fails a
  four-dimensional data space.
- D-190's gate is too narrow (the identity should be refused everywhere
  it is literal); D-168's acceptance of a whitened identity prior in
  tutorial 13 is superseded by D-190.
- D-183's naming (`dense=`), not its substance.

## Should-fix themes worth a pass each

- **Checks that pass what they should catch**: `check_measure` never
  touches the precision; `check_operator` never compares a supplied
  `components_action`; `check_traits` runs a dense rank decomposition for
  every diagonal or composed covariance claiming invertibility (n³).
- **Cost regressions**: `D + I` stays generic (the D-170 failure back by
  one `+`); the diagonal of a composition with a sparse factor densifies;
  Jacobi probes a diagonal metric column by column; `log_posterior.at`
  costs two linearizations; nonlinear blocks evaluate twice in `at()`;
  `support_sweep(n_jobs=)` results not kept; `plot_set` runs 180 serial
  solves with no `n_jobs=`; sampling routes hold every draw in memory.
- **Degenerate priors** (D-138 admits them, no algorithm can use them);
  `n_jobs=` missing on `inner_hull`, `extent`, `outer_ellipsoid`, the
  mixture's loops; `via=` forwarding; `contains` tolerances floored at one.
- **Surface**: five apps hand-roll the prior-norm whitening that
  `LeastSquares`/`MinimumNorm` could take as `covariance=`; `marginal`
  loses samplers and precisions; `from_product` drops precision factors;
  `AbsolutelyContinuousMeasure.can_sample` lies; `hilbert_schmidt_norm`
  and `log_normalizing_constant` return stochastic estimates as floats;
  `method="spectral"` means two routes; `HilbertSpace.mean` against both
  naming rules (v1: `sample_expectation`); `harden_error` public with only
  tests calling it; `Minimization.push_forward` returns a private copy.
- **Docstrings that misstate the code**: the default solver tolerance,
  the retired backus routes, `from_callables(linearize=, hessian=)`,
  CURRENT_STATE §7's dense point-evaluation entry (stale since D-183).

## Decisions taken with David (2026-10-29)

1. **Identity gate widened to every route** (literal identity: constant
   spectrum, identity or scaled-identity operator or factor, constant
   deviation, flat power), each with `unsafe=`; docstrings say the
   trace-class condition beyond that is the user's. Tutorials 9 and 13
   re-posed on trace-class priors with the prior covariance as the
   Newton-CG preconditioner; tutorial 9's optimizer argument rewritten.
   D-168's whitened identity prior is superseded by D-190.
2. **Both settled decisions stand.** The chi-squared potential keeps its
   internal bypass (a misfit is a potential); infinite-dimensional data
   spaces are not a concern of the library; `consistency_set` is made
   consistent with the potential, and both docstrings say the error
   measure must be trace class for a limit to exist. The model-space
   formalism exists for explicitly finite-dimensional problems and
   low-dimensional surrogates, which its docstrings now say; its
   evidence there is a direct determinant, and the mixture's docstring
   recommends that over the stochastic route. The reviewers'
   measurements go to CURRENT_STATE §7.
3. **Naming.** Matrix-free point evaluation: `route="auto"|"dense"|"local"`
   on both space families, `dense=` and `form=` gone from it. The
   symmetric-space names leave the top level (`Box`, `Interval`,
   `Lebesgue`, `PeriodicBox`, `Sobolev`) and live only in
   `symmetric_space`, as the spectral-element spaces do in theirs; the
   Fourier `Interval` factory stays there under its own name.
   `HilbertSpace.mean` becomes `sample_expectation`, no alias.
4. **Outer ellipsoid.** The model-space S-procedure becomes the default;
   the data-space form stays behind `formalism=` with its regime of
   validity documented; both check the ellipsoid against a few support
   values before returning, inflating or raising when it falls short, so
   the docstring's claim becomes true. The data-space route's repair
   through the resolvent identity and a floored walk is a post-release
   item in §7; a small API change later is acceptable.
5. **Examples and tutorials.** The four new examples and the gap-free
   renumbering as proposed; the three contradicting scripts fixed; the
   tutorial course reordered to point, measure, set with its citations
   followed through. Framing, from David: the library exposes point
   estimators because they are wanted, but its view is that they are
   tools within the proper methods with uncertainty quantification and
   not generally interesting in themselves; there will be no "optimal
   regularization parameter" or the like in this library. The new
   point-estimates example says so.

## Decisions taken with David (2026-10-30)

6. **Optimizers stop on relative tolerances.** The gradient tolerance
   relative to the gradient at the start with the absolute one kept as a
   user-settable floor, the precision floor relative to `|f|` at the
   start, and the Armijo minimum step, the slope-ratio clip and the
   proximal gradient's probe length made relative likewise; the nonlinear
   apps and tutorials re-measured.
7. **Definiteness is propagated, not stamped.** A push-forward `T C T*`
   of a definite covariance claims definiteness when `T` is surjective
   (an operator claiming invertibility or surjectivity); `credible_set`
   and `condition` then refuse a merely semidefinite covariance by name
   instead of claiming it. Tutorial 6's push-forwards pass through their
   surjective property operators.
