# pygeoinf Strategic Report — Strengths, Weaknesses, Opportunities, Risks, and a Road Forward

> Author: agentic survey, 2026-05-15.
> Scope: the [pygeoinf](/home/adrian/PhD/Inferences/pygeoinf) package only (not [intervalinf](/home/adrian/PhD/Inferences/intervalinf) or [pygeoinf3D](/home/adrian/PhD/Inferences/pygeoinf3D), but with explicit reference to them as exemplar submodules).
> Status: snapshot at v1.7.8 (poetry), branch `mission/20260403-lowering-master`.
> Audience: core devs (David Al-Attar, Dan Heathcote, Adrian Mag) preparing to position pygeoinf as a community-facing scientific library.

---

## How to read this report

This report is intentionally multiscale. Read it like a zoom lens.

1. **§1 Executive summary** — one page, the elevator pitch and the headline tradeoffs.
2. **§2 The user's mental model — 6 elements, 2 paths, 2 modes** — the unified picture every user should leave the homepage with: the 6 mathematical elements, the measure-vs-set paths, the inversion-vs-inference modes, the bridges between them, and the gap between this vision and the current code.
3. **§3 Vision check** — does the stated philosophy ("mathematics-first, discretization-agnostic, modular orchestrator, linear-problem-focused for now") match the code as it currently stands?
4. **§4 Architectural map** — bird's-eye view of how the modules fit together, with file references.
5. **§5 Strengths (SWOT)** — what is genuinely world-class and load-bearing for the pitch.
6. **§6 Weaknesses (SWOT)** — concrete frictions, gaps, and tech debt that would impede adoption.
7. **§7 Opportunities (SWOT)** — strategic levers and emerging directions.
8. **§8 Risks (SWOT)** — what could undermine adoption or fork the community.
9. **§9 Detailed module-by-module audit** — the small-scale view: file-level notes with line counts and quality flags.
10. **§10 Roadmap proposal** — phased plan (3 months, 6 months, 12 months) with concrete deliverables.
11. **§11 Open questions for the dev team** — things this report cannot resolve without a conversation.

**Scope.** This revision focuses on **linear inverse problems and inference** only. Non-linear work (`NonLinearOperator`, `ScipyUnconstrainedOptimiser`, etc.) is still present in the code and not deprecated, but the public story, the tutorials, and the first wave of polish all target the linear case.

**Vocabulary.** A **"submodule"** in the user's sense means a downstream library that supplies concrete `HilbertSpace`, `LinearOperator`, and `GaussianMeasure` implementations layered on top of pygeoinf core. The two existing exemplars are the in-repo [symmetric_space/](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/symmetric_space) subpackage (which lives inside pygeoinf for historical reasons) and the external [intervalinf](/home/adrian/PhD/Inferences/intervalinf) and [pygeoinf3D](/home/adrian/PhD/Inferences/pygeoinf3D) packages.

---

## 1. Executive summary

**Headline thesis.** pygeoinf is, in 2026, one of the cleanest expressions of *coordinate-free inference* available in the Python scientific stack. The abstractions are mathematically honest: a `HilbertSpace` knows about its inner product via the Riesz map; a `LinearOperator` knows its adjoint and dual; a `GaussianMeasure` is a measure on an abstract space rather than a wrapper around a covariance matrix; convex sets are first-class objects represented via their support functions. **No competing Python package combines this level of mathematical fidelity with usable, tested code.** The pitch — "write down the mathematics, get working code" — is real.

**However**, the package is currently in a posture optimised for a small, expert development group, not for community uptake:

- The README sells finished features but understates the *philosophical commitment* that distinguishes pygeoinf from PyTorch+SciPy or hIPPYlib.
- The documentation surface (Sphinx site, tutorials, agent-docs) is partial, inconsistent, and (deliberately) frozen for AI-assisted devs rather than human newcomers.
- The "submodule" pattern is only weakly formalised: `symmetric_space/` lives inside pygeoinf, while sibling packages live outside it; there is no contract, no template, and no extension registry.
- Some modules have grown well past 1k lines (e.g. [convex_optimisation.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_optimisation.py) at 2601, [plot.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/plot.py) at 2098, [gaussian_measure.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/gaussian_measure.py) at 2075) and now mix beginner-facing and expert-facing surface.
- No performance / scaling story is on the public page yet, and there is a live execution-planning mission (`mission/20260403-lowering-master`) that may reshape the operator semantics.

**The strategic question** for the team is therefore not *"is the code good enough?"* (it is) but *"are we ready to be opinionated about who the user is, what a submodule is, what workflow they pick, and what we promise to support?"*. The single highest-leverage move is to **publish a short, ruthlessly opinionated "pygeoinf manifesto + workflows + submodule contract"** alongside the existing README. Everything else in this report serves that move.

**Headline organising principle (see §2 for the full treatment).** A linear inverse problem in pygeoinf is built from up to **six mathematical elements** (model space, data space, property space, forward mapping, property mapping, prior information) which the user combines along **two paths** (measure-based with a `GaussianMeasure` prior, or set-based with a convex `Subset` prior) in **two modes** (inversion to recover the model, or inference to recover a property of the model). That is a 2×2 matrix of named workflows:

|                | **Inversion** — recover u ∈ M | **Inference** — recover q = T(u) ∈ P |
|----------------|-------------------------------|--------------------------------------|
| **Measure path** | `LinearBayesianInversion` ✓ | `LinearBayesianInference` (gap)      |
| **Set path**     | `LinearSetInversion` (gap)  | `LinearSetInference` (BG/DLI; not named as such) |

Three of the four cells are not yet *named*, *typed*, or *advertised* as such — even though the underlying code largely exists. This is the single most impactful gap to close before any wider promotion.

**Top six concrete recommendations** (detailed in §10):

1. **Ship the 2×2 matrix.** Introduce explicit `Likelihood` and `SetLikelihood` objects, a `LinearInference` parallel to `LinearInversion`, and the missing classes (`LinearBayesianInference`, `LinearSetInversion`, `LinearSetInference`). Surface the existing measure→set bridge (`credible_set` aka *hardening*) as an explicit cross-path operation.
2. Write a 1–2 page **"Manifesto + 2×2 workflows + submodule contract"** that pins the philosophy, names the four workflows and the bridges between them, and is testable (a `pygeoinf.testing` compliance pack that any submodule can run).
3. Carve the `symmetric_space/` subpackage out of pygeoinf into its own repo (e.g. `pygeoinf-symmetric`) to make it the **exemplar third-party submodule**.
4. Stabilise the public API in `pygeoinf/__init__.py` and *only* what's there; demote internal helpers to `pygeoinf._internal` or private modules so v2.x can refactor without breaking users.
5. Refactor the three "mega-modules" (`convex_optimisation`, `plot`, `gaussian_measure`) into curated subpackages with clear beginner-facing vs. expert-facing layers.
6. Adopt a "user persona" approach in the docs: a **fast-path tutorial per workflow** ("solve a measure-path Bayesian inversion in 30 lines"; same for the other three cells), a **submodule-author tutorial** ("wrap your own forward model on top of a function space"), and a **theory companion** that mirrors the code.

---

## 2. The user's mental model — 6 elements, 2 paths, 2 modes

This is the headline organising principle for the rest of the report and (eventually) for the public-facing materials. **If the marketing site has one diagram, this is it.** Every later weakness, opportunity, and roadmap step refers back to this section.

### 2.1 The six mathematical elements of a linear inverse problem

A user comes to pygeoinf with a problem in mind. To set it up they pick from this small fixed list:

| # | Element | Type | Required for inversion? | Required for inference? |
|---|---------|------|-------------------------|-------------------------|
| 1 | **Model space** M | `HilbertSpace` (typically a function space supplied by a submodule) | ✓ | ✓ |
| 2 | **Data space** D | `HilbertSpace` (typically finite-dim, `EuclideanSpace(m)`) | ✓ | ✓ |
| 3 | **Property space** P | `HilbertSpace` (often `EuclideanSpace(k)` for a handful of scalar properties) | — | ✓ |
| 4 | **Forward mapping** G : M → D | `LinearOperator` | ✓ | ✓ |
| 5 | **Property mapping** T : M → P | `LinearOperator` | — | ✓ |
| 6 | **Prior information** on M | `GaussianMeasure` (measure path) **or** convex `Subset` (set path) | ✓ | ✓ |

The user's workflow is uniform: **pick a submodule that supplies M (a function space implementation), wrap your own physics into a `LinearOperator` G : M → D, add a noise model, add prior information on M, optionally add T and P, and pick a workflow.** All the rest is plumbing pygeoinf does for you.

### 2.2 The two paths — measure vs. set

The way prior information and noise enter the problem defines two parallel paths through the library. They are mathematically symmetric and should be presented as such.

**Measure path (probabilistic).** The user equips M with a Gaussian prior measure μ_M and D with a Gaussian noise measure μ_E. Together with the forward operator G these *define a likelihood* — the conditional distribution of d given u. The combination of prior and likelihood is the standard linear-Gaussian Bayesian setup. The output is a Gaussian *posterior measure* on M (inversion) or its push-forward onto P (inference).

**Set path (set-theoretic / convex).** The user equips M with a convex prior set B ⊆ M (represented by its `SupportFunction` σ_B) and D with a convex error set V ⊆ D (represented by σ_V). Together with G these *define a set-likelihood* — the set-valued map u ↦ G(u) + V that returns the set of data values consistent with model u. Combined with observed data d̃ and the prior set B, this carves out an **admissible region** in M (inversion) or in P (inference). The output is a `Subset` (or, in the inference mode, an interval / convex hull in P).

> Side note. The two paths are not arbitrary alternatives. They correspond to two genuinely distinct epistemic stances: "I believe the prior probability distribution" vs. "I am only willing to bound the prior". Pygeoinf is one of the very few libraries to take both stances first-class.

### 2.3 The two modes — inversion vs. inference

This is the second axis. It is orthogonal to the path axis.

**Inversion mode.** The user wants to recover u itself (or its distribution / admissible region) in M. They supply elements 1, 2, 4, 6.

**Inference mode.** The user wants to recover a *property* of u — a linear functional, or a vector of linear functionals — in some property space P, without committing to a recovery of u itself. They additionally supply elements 3 and 5. This is the Backus-Gilbert worldview: "I cannot resolve the full model, but I can give you defensible bounds (or a posterior distribution) on this particular property of it."

For users coming from the Bayesian world, inference is "the posterior on T(u), reported via P". For users coming from the geophysics / classical-inverse-problems world, inference is "the resolvable parameters, and what we can say about them with the data we have".

### 2.4 The 2×2 matrix of named workflows

Cross the path axis with the mode axis and we get four named workflows. **Each should be a first-class class in the public API.**

|                                           | **Inversion mode** — output lives in M | **Inference mode** — output lives in P |
|-------------------------------------------|----------------------------------------|----------------------------------------|
| **Measure path** (Gaussian prior + likelihood) | `LinearBayesianInversion` (Gaussian posterior measure on M) | `LinearBayesianInference` (Gaussian posterior on P = T#posterior_on_M) |
| **Set path** (convex prior + set-likelihood)   | `LinearSetInversion` (admissible region in M) | `LinearSetInference` (admissible interval/region in P; Backus-Gilbert / DLI) |

**Mapping to the current code:**

- *Measure × inversion.* `LinearBayesianInversion` ✓ — exists and is well-developed ([linear_bayesian.py:55-1154](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_bayesian.py#L55)). The dual `formalism="model_space"|"data_space"` switch is unusually well-thought-out.
- *Measure × inference.* **Gap.** The posterior measure can be pushed forward onto P by `posterior_measure.affine_mapping(operator=T)`, but there is no class that takes (forward_problem, model_prior, property_operator, property_space, data) and presents the property posterior with the same convenience surface (`expectation_operator`, `kalman_operator` analogue, sampler) as the inversion case. The user is expected to assemble this manually.
- *Set × inversion.* **Gap.** The constrained least-squares / minimum-norm classes (`ConstrainedLinearLeastSquaresInversion`, `ConstrainedLinearMinimumNormInversion`) act on affine subspace constraints — that is a *set inversion* with a particular kind of prior set, but it is not framed in those terms and the more general "convex prior set + convex error set" case has no dedicated class. The machinery to do it exists (support functions, convex subsets, `PrimalKKTSolver`, `ProximalBundleMethod`) but the user-facing wrapper is missing.
- *Set × inference.* **Partly there.** The `DualMasterCostFunction` ([backus_gilbert.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/backus_gilbert.py)) and the sphere DLI example ([work/sphere_dli_example.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/work/sphere_dli_example.py)) implement this workflow end-to-end. But the user-facing class is named after the *algorithm* (Backus-Gilbert dual master cost) rather than after the *workflow* (set inference). A `LinearSetInference(forward_problem, model_prior_set, data_error_set, property_operator, property_space, data)` wrapper that picks the right solver and returns admissible bounds is the natural next step.

So **the structural code is largely there; what's missing is the framing, the naming, and a small number of orchestrator classes**. This is a low-effort, high-leverage opportunity.

### 2.5 The bridges between paths and modes

The matrix is more powerful when the cells are connected. Pygeoinf should expose two kinds of bridge as first-class operations.

**Path bridges (measure ↔ set).**

- **Hardening: measure → set.** Given a Gaussian (prior or posterior) measure μ on a space, produce a convex set at a chosen probability level. *Already implemented* as `GaussianMeasure.credible_set(probability, geometry=…)`, with three geometries (`ellipsoid`, `ambient_ball`, `weakened_ellipsoid`) — see [convex-analysis-reference.md:77-99](/home/adrian/PhD/Inferences/pygeoinf/docs/agent-docs/references/living/convex-analysis-reference.md#L77-L99). This is the canonical bridge from the measure path to the set path. The docs do not currently *name* it as such.
- **Softening: set → measure.** Given a convex prior set B, produce a Gaussian prior whose credible region at some probability matches B. Less canonical than hardening (many measures produce a given set), but useful for sensitivity analyses, warm-starts, and turning a set-inversion result into a Bayesian posterior for downstream propagation. **Should be specified and implemented, even if just for a small set of natural choices (ellipsoid → matched Gaussian, ball → isotropic Gaussian).**

**Mode bridges (inversion ↔ inference).**

- **Inversion → inference.** Given the result of an inversion (posterior measure on M, or admissible region in M) and a property operator T : M → P, push it forward to P. *Trivial for measures* (affine_mapping), *well-defined for sets when T is linear* (image of a convex set under a linear map is convex — already implemented as `LinearImageSupportFunction`, but not exposed as a one-line method on the inversion result objects). **Should be a one-liner: `inversion_result.infer(property_operator, property_space)` → `InferenceResult`.**
- **Inference → inversion.** Generally ill-posed (a property is a quotient of the model; you can't recover the model from one of its quotients), but worth a documented note pointing users at the right interpretation.

The four cells of the matrix plus the four bridges give the user a complete map. *Every line they write should correspond to either picking a cell, or invoking a bridge.*

### 2.6 What an idiomatic user session should look like

Two examples, fictionalised but close to what the code can already do (or could after the small additions in §10).

**Measure path, inversion mode** — Gaussian Bayesian inversion in ~12 lines:

```python
import pygeoinf as pgi
from pygeoinf_symmetric.sphere import Sobolev      # hypothetical post-carve-out name

# 1, 2, 4: spaces and forward operator
M = Sobolev(lmax=64, order=1.5)
D = pgi.EuclideanSpace(m=500)
G = build_my_forward_operator(M, D)                  # user's own physics

# 6: prior and noise
prior = M.heat_kernel_prior(scale=0.1)               # Gaussian on M
noise = pgi.GaussianMeasure.from_standard_deviation(D, 0.02)
fp = pgi.LinearForwardProblem(G, data_error_measure=noise)

# Pick the cell. Solve. Inspect.
inv = pgi.LinearBayesianInversion(fp, prior)
posterior = inv.model_posterior_measure(d_observed, solver=pgi.CGSolver())
print(posterior.expectation, posterior.covariance)
hardened = posterior.credible_set(probability=0.95)  # measure → set bridge
```

**Set path, inference mode** — Backus-Gilbert admissible region for a property, in ~14 lines:

```python
# 1, 2, 4 as above; plus property space and property operator
P = pgi.EuclideanSpace(k=6)
T = build_cap_average_property_operator(M, P, target_caps)

# 6 (set version): convex prior + convex noise set
prior_set = pgi.BallSupportFunction(M, center=M.zero, radius=truth_norm_bound)
noise_set = pgi.BallSupportFunction(D, center=D.zero, radius=3*sigma*np.sqrt(D.dim))

# Pick the cell. Solve. Inspect.
inf = pgi.LinearSetInference(fp, prior_set, noise_set, T, P)        # NEW class
bounds = inf.admissible_region(d_observed, solver=pgi.PrimalKKTSolver(...))
print(bounds.lower, bounds.upper)
```

The clarity of these examples is the single best measure of whether the library is ready for promotion. **Right now the measure-inversion example is essentially correct; the set-inference example requires the user to construct a `DualMasterCostFunction` per property direction by hand.** Closing this gap is Horizon-1 work.

### 2.7 What is missing in the current code to fully realise this picture

Concrete and small. Listed for action in §10.

1. **No first-class `Likelihood` object.** The pair (G, μ_E) implicitly defines a Gaussian likelihood; the pair (G, V) implicitly defines a set-likelihood. Neither is a class today. Promoting them gives users a vocabulary, makes the 2×2 symmetric, and matches the textbook.
2. **No `LinearInference` parent class** to mirror `LinearInversion`. Should hold (forward_problem, property_operator, property_space) and provide the common scaffolding.
3. **Missing `LinearBayesianInference` class.** Cell (measure, inference) of the matrix.
4. **Missing `LinearSetInversion` class.** Cell (set, inversion) of the matrix. (The constrained linear classes are special cases but should remain as such.)
5. **`LinearSetInference` does not exist as a named user-facing class.** The functionality is split between `DualMasterCostFunction` and a hand-rolled per-direction loop in `work/sphere_dli_example.py`. Promote it.
6. **The bridges are not called bridges.** `GaussianMeasure.credible_set` is the hardening bridge but isn't documented under that label. There is no documented softening bridge. The inversion→inference push-forward is implicit.
7. **`LinearForwardProblem` is the wrong granularity for the set path.** It hardcodes a `GaussianMeasure` data error. A more uniform abstraction would let it carry either a `GaussianMeasure` *or* a convex `Subset` as the data error — either via two siblings or via a generic `data_error: Union[GaussianMeasure, Subset]` field.
8. **The README does not show this map.** This is the marketing fix, but it's downstream of the code fixes 1-7 above.

These additions are *re-organisations and small wrappers*, not algorithmic work. They can be delivered in a few weeks if treated as a priority.

### 2.8 Why this framing matters for the manifesto

The user reading the README should be able to say, within five minutes:

> "Oh, I can use actual function spaces! And I can wrap my own forward mapping on one of their operators and link it to the function space implemented in some submodule. Then I pick: measure path or set path? Then I pick: do I want the model itself, or a property of it? Each of those four choices is a named class. And the docs tell me how to convert between the choices once I've made one."

That sentence is the manifesto. The rest of the report is about making it true.

---

## 3. Vision check — does the code match the pitch?

The stated brand promise has three pillars. Let's score each against the current code, with evidence.

### 3.1 "Mathematical rigor" — Pillar A

**Score: 9/10. Strong.**

The Hilbert space abstraction is built around the Riesz map, *not* around a particular component representation. Evidence:

- [hilbert_space.py:51-446](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/hilbert_space.py#L51-L446) — `HilbertSpace` requires `to_dual`/`from_dual` as abstract methods; the inner product is *derived* from them (line 170-184). A subclass cannot accidentally lie about its inner product because the duality test ([checks/hilbert_space.py:58-73](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/checks/hilbert_space.py#L58-L73)) is a mixin axiom check that any concrete space must pass.
- [linear_operators.py:46-1037](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_operators.py#L46-L1037) — `LinearOperator` carries both an `adjoint` (uses the inner product) *and* a `dual` (lives in `codomain.dual -> domain.dual`). Few libraries make this distinction; it is the one that survives changes of inner product and matters in PDE-constrained inversion.
- [linear_operators.py:131-220](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_operators.py#L131-L220) — `from_formal_adjoint` / `from_formally_self_adjoint` correctly lift operators between `MassWeightedHilbertSpace` instances. This is precisely the trick FEM users need; it is non-trivial and rarely seen in pure-Python form.
- [gaussian_measure.py:63-150](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/gaussian_measure.py#L63-L150) — A `GaussianMeasure` is parameterised by an operator-valued covariance, not by an array. It supports `covariance_factor`, sampling via Cholesky/eigendecomposition, KL divergence in operator form, and affine pushforwards. Compare to `scipy.stats.multivariate_normal`, which is array-only.
- [convex_analysis.py:15-150](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py#L15-L150) — Convex sets are represented by *support functions* with optional support-point oracles. This is the right abstraction for non-Gaussian inference (Backus-Gilbert, robust inversion, deterministic uncertainty quantification) and is unusually principled.

**One blemish.** Some classes still leak the component dimension into definitions where they shouldn't (e.g. `HilbertSpace.dim` is `abstractmethod` and required to be finite at line 67-69). This is fine for the current implementations but pre-commits pygeoinf to discrete-after-truncation. The discretization-agnostic promise (pillar B) is *operationally* honoured but the **type system says "I am finite-dim"** the moment a concrete space is built. This is worth being explicit about — see §6.

### 3.2 "Discretization agnostic" — Pillar B

**Score: 8/10. Strong but with caveats.**

The architecture genuinely allows the same algorithm code to run on different discretizations:

- A `LinearLeastSquaresInversion` ([linear_optimisation.py:46-200](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_optimisation.py#L46-L200)) constructs its normal equations using `forward_operator.adjoint`, `data_error_measure.inverse_covariance`, and `+` on operators. None of those calls reach into component arrays. The same class runs unchanged whether `model_space` is `EuclideanSpace(n)`, `symmetric_space.sphere.Sobolev(...)`, or an [intervalinf.spaces.Lebesgue](/home/adrian/PhD/Inferences/intervalinf) instance.
- Direct-sum spaces ([direct_sum.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/direct_sum.py)) compose without special-casing.
- Solvers come in two flavours: *matrix-based* (LU, Cholesky, GMRES on the dense Galerkin matrix) and *matrix-free* (CG, MinRes, BICGStab on the abstract operator). See [linear_solvers.py:40-120](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_solvers.py#L40-L120).

**Caveats:**

- Every concrete space must currently materialise an `np.ndarray` of components (the `to_components`/`from_components` round-trip is mandatory at [hilbert_space.py:100-122](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/hilbert_space.py#L100-L122)). This means truly meshless / lazy representations are awkward to plug in. The Lowering Execution mission appears to be reshaping this.
- "Galerkin vs standard" matrix representation ([linear_operators.py:569-665](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_operators.py#L569-L665)) is correct but the docstrings repeatedly warn that picking the wrong one silently produces a different operator. Users need to internalise this distinction or they will produce garbage.
- The `EuclideanSpace` is the default fallback everywhere (line 232, 271, 314, etc. — coordinate inclusion/projection always lands in `EuclideanSpace(self.dim)`). A user who needs a *non-finite-coordinate* abstract space has to think hard.

### 3.3 "Modular, orchestrator-first" — Pillar C

**Score: 5/10. Aspirational, not yet operational.**

This is the weakest pillar in the current code, and the one with the most room for improvement. (The closely related gap — the 2×2 workflow matrix not being surfaced — is §2.7 above.)

- **What's already modular:** `symmetric_space/` is an in-tree subpackage with its own `__init__.py`, lazy-imports its concrete classes, and provides a clean recipe (an abstract `SymmetricHilbertSpace` ABC implementing all the heavy lifting; concrete spaces like `sphere.Lebesgue` inherit and supply only basis-specific bits). See [symmetric_space/__init__.py:7-33](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/symmetric_space/__init__.py#L7-L33) and [sphere.py:57-103](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/symmetric_space/sphere.py#L57-L103).
- **What's not yet modular:**
  - `symmetric_space` lives *inside* pygeoinf rather than being a separate distribution. So pygeoinf depends (optionally, via the `sphere` extra) on `pyshtools` and `Cartopy`. That's a hard ergonomic block: any submodule author looking for a template will copy this pattern and either bloat their package or vendor pygeoinf.
  - There is **no published contract** for what a "submodule" must implement. A would-be author has to read the Hilbert space ABC, the axiom checks, and the operator factory methods to reverse-engineer the contract.
  - There is **no registry / discovery mechanism**. Users can't `pip install pygeoinf-fenics` and have it advertise itself.
  - The README's "convenience methods that wrap a lot of complex code" is *only weakly demonstrated* by current concrete classes (e.g. `sphere.Lebesgue.from_covariance` at [sphere.py:104-150](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/symmetric_space/sphere.py#L104-L150) is a good example, but those are scattered, not visible as a deliberate pattern). The 2×2 matrix from §2 gives a natural home for these: a submodule should ship one-line constructors for the priors that are natural in its space (heat-kernel Gaussian priors, ball/ellipsoid prior sets calibrated to typical model norms, etc.).
  - The boundary between *what should live in core* and *what is a submodule concern* is fuzzy: `datasets.py` (GSN/USGS), `plot.py` (cartopy-dependent in places), and `data_assimilation/` are all in core but feel more like submodule-flavour content.

This is the second biggest gap between vision and execution (after the 2×2 matrix gap in §2). §7 and §10 propose concrete actions.

---

## 4. Architectural map (bird's-eye)

```
                              ┌─────────────────────────────┐
                              │   USER FACING ENTRY POINTS  │
                              ├─────────────────────────────┤
                              │  LinearBayesianInversion    │
                              │  LinearLeastSquaresInversion│
                              │  LinearMinimumNormInversion │
                              │  Constrained{…}             │
                              │  DualMasterCostFunction (BG)│
                              └────────────┬────────────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
   ┌────────▼─────────┐         ┌──────────▼──────────┐        ┌──────────▼──────────┐
   │ ForwardProblem,  │         │ LinearSolvers       │        │ Convex {analysis,   │
   │ LinearForward    │         │ Preconditioners     │        │ optimisation},      │
   │ Problem          │         │ LowRank (SVD/Chol)  │        │ Subsets, Subspaces  │
   └────────┬─────────┘         └──────────┬──────────┘        └──────────┬──────────┘
            │                              │                              │
            └──────────────────────────────┼──────────────────────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   OPERATOR ALGEBRA       │
                              │  LinearOperator (+adj    │
                              │  +dual +matrix          │
                              │  +block/direct sum)      │
                              │  AffineOperator,         │
                              │  NonLinearOperator       │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │      HILBERT LAYER       │
                              │  HilbertSpace (ABC)      │
                              │  DualHilbertSpace        │
                              │  MassWeightedHilbertSpace│
                              │  HilbertSpaceDirectSum   │
                              │  EuclideanSpace          │
                              │  LinearForm/NonLinearForm│
                              │  GaussianMeasure         │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │     AXIOM CHECKS         │
                              │  checks/{hilbert_space,  │
                              │   linear_operators,      │
                              │   nonlinear_operators,   │
                              │   affine_operators}      │
                              └──────────────────────────┘

      ───────────────────────────────────────────────────────────────────
                          (boundary that today is fuzzy)
      ───────────────────────────────────────────────────────────────────

   IN-TREE CONCRETE LAYER             EXTERNAL SUBMODULES (sibling pkgs)
   ┌─────────────────────────┐        ┌──────────────────────────────┐
   │ symmetric_space/        │        │ intervalinf  (1D interval)    │
   │   line, circle, sphere, │        │ pygeoinf3D   (3D box domains) │
   │   torus, plane, ABC     │        │ simple_regional_tomography…   │
   │ datasets/ (GSN, USGS)   │        │ (… your future submodule …)   │
   │ data_assimilation/      │        │                                │
   │   pendulum demo         │        │                                │
   │ plot.py (matplotlib +   │        │                                │
   │   cartopy slices)       │        │                                │
   └─────────────────────────┘        └──────────────────────────────┘
```

### 4.1 The 33-module inventory

The `pygeoinf/` package contains 33 Python modules. Grouped by role:

| Role | Modules |
|------|---------|
| Hilbert layer | `hilbert_space.py` (850), `linear_forms.py` (263), `nonlinear_forms.py` (340), `direct_sum.py` (534) |
| Operator algebra | `linear_operators.py` (1602), `nonlinear_operators.py` (219), `affine_operators.py` (174), `spectral_operator.py` (261), `matrix_function.py` (216) |
| Probability | `gaussian_measure.py` (2075), `quadratic_form_quantile.py` (526) |
| Sets and geometry | `subspaces.py` (799), `subsets.py` (1713), `convex_analysis.py` (923) |
| Solvers and optimisation | `linear_solvers.py` (1053), `preconditioners.py` (549), `low_rank.py` (986), `linear_optimisation.py` (1330), `linear_bayesian.py` (1154), `nonlinear_optimisation.py` (218), `convex_optimisation.py` (2601) |
| Inverse-problem orchestration | `forward_problem.py` (432), `inversion.py` (312), `backus_gilbert.py` (399) |
| Concrete spaces | `symmetric_space/` (line, circle, sphere, torus, plane, plus ABC) |
| Auxiliary | `datasets.py` (226), `plot.py` (2098), `parallel.py` (73), `utils.py` (15), `config.py` (9), `auxiliary.py` (28), `dynamical_system.py` (111), `data_assimilation/` |
| Axiom checks | `checks/` (mixin axiom-test suites for each abstract base) |

Total: **~22.3k lines** in the package, **~14k lines** in 60 test files, **12 tutorial notebooks**, and **~13k lines** of agent-docs.

### 4.2 Internal couplings worth noting

- `hilbert_space.py` is leaf-ish (only depends on `checks/hilbert_space.py`). Good.
- `linear_operators.py` imports `nonlinear_operators.py` and `parallel.py`, and pulls in `direct_sum.py` lazily inside methods. Cycles are avoided with `TYPE_CHECKING` guards (good practice).
- `gaussian_measure.py` is dependency-heavy: it imports from `hilbert_space`, `linear_operators`, `linear_solvers`, `affine_operators`, `direct_sum`. This is correct but makes the module the *most fragile* refactor target.
- `linear_bayesian.py` depends on `inversion`, `gaussian_measure`, `forward_problem`, `linear_operators`, `linear_solvers`, `hilbert_space`, `affine_operators`, `low_rank` — eight imports. This concentrates a lot of integration risk in one file.
- `convex_optimisation.py` depends on `nonlinear_forms` and `convex_analysis` only; that is well-scoped. But the module itself is 2601 lines and contains multiple algorithms (`SubgradientDescent`, `ProximalBundleMethod`, `PrimalKKTSolver`, plus `KKTResult`). It is overdue for splitting.
- `plot.py` (2098 lines) mixes generic posterior-corner plotting, 1D distributions, subspace slicing, and cartopy maps. This is the second clear refactor candidate.

---

## 5. Strengths (SWOT)

### S1. Mathematical fidelity is genuinely rare

There is no peer Python library that combines:

1. Coordinate-free Hilbert space ABCs with Riesz-map-defined inner products,
2. Operator algebra distinguishing adjoint from dual,
3. Mass-weighted spaces lifted via `from_formal_adjoint`,
4. Gaussian measures as covariance-operator objects with KL/credible-set machinery,
5. Convex sets via support functions with `image`/Minkowski-sum algebra,
6. Backus-Gilbert dual master cost as a NonLinearForm.

`hIPPYlib` and `MUQ` come closest but are heavier (FEniCS-bound; C++-bound) and less "Python-fluent". `cuqipy` is closer in spirit but does not have the convex-analysis surface or the dual-master cost machinery. **This is the pitch.** If you do nothing else, double down on this in the public-facing materials.

### S2. The 2×2 matrix is *almost* there in code

The architecture of §2 is not aspirational — it is what the code already does, *minus the framing*. Three of the four workflow cells (measure-inversion, set-inference, all the constrained variants of set-inversion) have working implementations. The two existing inversion tutorials, the sphere DLI example with its 13 tests, and `LinearBayesianInversion`'s polished dual-formalism switch all demonstrate this. Reorganising what exists is *much cheaper* than building from scratch — the strategic upside is unusually high for the cost.

### S3. The axiom-check framework

The [checks/](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/checks) subpackage is a hidden gem. Every abstract base class is paired with a mixin that, when run on a concrete implementation, samples random vectors and verifies algebraic identities (vector-space axioms, inner-product linearity/symmetry, triangle inequality, Riesz consistency, adjoint identity, dual-vs-component round-trip). This is what makes the modularity promise *operationally credible* — a submodule author can write a custom space and run `space.check()` to get immediate feedback. It is also exactly the right primitive to publish as a compliance test pack (see §10).

### S4. Tutorials exist and ladder up well

Tutorials 1–10 cover the conceptual progression — first example, Hilbert spaces, dual spaces, linear operators, solvers, Gaussian measures, MN/LSQ, Bayesian, direct sums, symmetric spaces — in a sequence that genuinely teaches. The Colab links lower the friction of "try it now". This is a competitive advantage over MUQ and hIPPYlib, which require a non-trivial local build before any tutorial runs.

### S5. Performance hooks are present where they matter

- `LinearOperator.matrix()` returns a *matrix-free* `scipy.sparse.linalg.LinearOperator` by default, with `dense=True` available; the dense path is parallelised over columns via `joblib` ([linear_operators.py:629-665](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_operators.py#L629-L665)).
- Randomised SVD / Cholesky / eigendecomposition are in `low_rank.py` and use the abstract operator interface, so they work on any submodule's operators without modification.
- Specialised sparse / diagonal-sparse operator classes give 10–100× wins on the right problems.
- The `InvariantLinearAutomorphism` on a `SymmetricHilbertSpace` is `DiagonalSparseMatrixLinearOperator`-backed, so its inverse / square root / spectral algebra is O(n) not O(n³). Same is true of `InvariantGaussianMeasure` (KL divergence runs in O(N) time, [symmetric_space.py:340-366](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/symmetric_space/symmetric_space.py#L340-L366)).

### S6. Test coverage and CI hygiene

- 60 test files, 233 `test_*` functions, ~14k lines of test code. The pytest run snapshot (`pytest_warnings.txt`) shows 494 collected items and the symmetric-space tests pass on Python 3.12.
- CI runs on Ubuntu, macOS, Windows × Python 3.12, with `ruff` linting, `pytest --cov`, and `poetry build` ([.github/workflows/ci.yml](/home/adrian/PhD/Inferences/pygeoinf/.github/workflows/ci.yml)).
- A publish workflow auto-pushes tagged releases to PyPI.

### S7. Bayesian dual-formalism is a competitive advantage

`LinearBayesianInversion(..., formalism="data_space" | "model_space")` is unusual and *exactly the right knob* for geophysics, where the data dimension is often much smaller than the model dimension. Few libraries expose this cleanly; pygeoinf does ([linear_bayesian.py:55-130](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_bayesian.py#L55-L130)). The `kalman_operator` factory takes a `LinearSolver` and an optional preconditioner, so users can choose their inversion strategy at the call site.

### S8. Backus-Gilbert is differentiated and topical — and is the set-inference workflow under a different name

The `DualMasterCostFunction` ([backus_gilbert.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/backus_gilbert.py)) implements the Hilbert-form dual master cost for support-function-based admissible regions. This is a recent (post-2020) line of research that very few libraries support. Coupled with the `sphere_dli_example` work-script (15 tests), pygeoinf has a story to tell about *deterministic uncertainty quantification* that competing packages do not.

### S9. Submodule template already exists (just unnamed)

The `symmetric_space.AbstractSymmetricLebesgueSpace` + `SymmetricSobolevSpace` pattern is exactly the recipe a submodule author should follow:

1. Subclass `HilbertSpace` (or `HilbertModule` if pointwise product is needed).
2. Implement `to_components`, `from_components`, `to_dual`, `from_dual`, `__eq__`, and (for `HilbertModule`) `vector_multiply`/`vector_sqrt`.
3. Provide a few "convenience constructors" (`from_covariance`, `from_heat_kernel_prior`, …).
4. Provide invariant operator and measure subclasses.
5. Optionally provide plotting integration.

This is a small contract and the existence proof is in the repo. The only missing step is *documenting* it as **the** contract.

---

## 6. Weaknesses (SWOT)

### W0. The 2×2 workflow matrix is not surfaced

The single most important weakness, in terms of the gap between user-facing experience and underlying capability. The full case is made in §2; in summary:

- **No `Likelihood` class** (measure path) and **no `SetLikelihood` class** (set path). The library has all the parts (forward operator, noise measure, support functions) but not the abstraction that says "together these define the conditional model of data given u".
- **No `LinearInference` parent class** to mirror `LinearInversion`.
- **Missing `LinearBayesianInference` and `LinearSetInversion` user-facing classes.**
- **`LinearSetInference` exists in code (the `DualMasterCostFunction` + per-direction loop pattern in `sphere_dli_example.py`) but is not named, packaged, or documented as a workflow.**
- **The hardening bridge (`GaussianMeasure.credible_set`) is not framed as a path-bridge in the docs.** No softening bridge exists.
- **`LinearForwardProblem` hardcodes a `GaussianMeasure` data error**, which makes it a clumsy fit for the set path.

These are small, targeted code additions and re-namings — but they have to happen *before* public promotion, because they are exactly what makes the manifesto's claim "you can choose between four named workflows" true.

### W1. The "submodule contract" is implicit

Right now an external author has to read at least:

- `hilbert_space.py` (the abstract methods)
- `checks/hilbert_space.py` (the axiom checks)
- One concrete reference (e.g. `symmetric_space/sphere.py`)
- The `from_components` invariants required by every internal algorithm

…to know what to implement. There is no `pygeoinf.subpackage_contract` module, no `SubmoduleCompliance` test class, no template repo. The promise of "any researcher can plug in their own discretization" is currently an ad-hoc reverse-engineering exercise.

**Fix:** publish a single `docs/source/submodule_author_guide.rst` and a `pygeoinf.testing` package that exposes `assert_hilbert_space_compliant(space)`, `assert_operator_compliant(op)`, `assert_gaussian_measure_compliant(mu)`. See §10.

### W2. Three "mega-modules" carry too much weight

| Module | Lines | Issue |
|--------|-------|-------|
| `convex_optimisation.py` | 2601 | Three distinct algorithms (subgradient, proximal bundle, primal KKT) + result types in one file. The QP-backends story is mixed in. |
| `plot.py` | 2098 | Mixes 1D distribution plots, corner plots, subspace slicing, and symmetric-space map projections. |
| `gaussian_measure.py` | 2075 | Constructor maze (`from_standard_deviation`, `from_factor`, `from_samples`, `from_direct_sum`, `from_function_space`, …), credible-set machinery, KL divergence, low-rank approximations, all in one file. |

Each of these would benefit from being a subpackage: `pygeoinf.convex_optimisation.{subgradient,bundle,primal_kkt}`, `pygeoinf.plot.{distributions,corner,slice,maps}`, `pygeoinf.gaussian_measure.{core,constructors,credible_sets,low_rank}`. The benefit is twofold: it scopes user attention ("import the bundle solver, not the proximal grab-bag"), and it makes substitution easier (a submodule author can plug in their own `subgradient` backend without touching the rest).

### W3. The public API is implicit and over-broad

[pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py) exports **75 symbols** in `__all__`. Many of these are implementation details (`MatrixLinearOperator`, `DenseMatrixLinearOperator`, `BICGMatrixSolver` vs `BICGStabMatrixSolver`, `JacobiPreconditioningMethod` and six other preconditioners, `BlockStructure`, …). This makes every refactor a potential breaking change.

**Fix:** classify symbols into three tiers — `core` (stable, semver-protected), `advanced` (stable for power-users, may evolve), `experimental` (no stability promise). Put this distinction in the docs and reflect it via subpackage organisation.

### W4. Documentation is incomplete and out of date in places

- The Sphinx site source ([docs/source/](/home/adrian/PhD/Inferences/pygeoinf/docs/source/)) is bare: `index.rst` repeats the README, then defers to `modules.rst` for autodoc. There is no User Guide, no "How do I…?" cookbook, no theory primer. RST → HTML still produces something usable because docstrings are good, but no human will discover the *philosophy* of the library from the docs.
- The agent-docs structure (`docs/agent-docs/`) is sophisticated and well-maintained but **explicitly written for AI collaborators**. A new contributor reading it will think pygeoinf is a closed shop.
- Two top-level files (`CONVEX_ANALYSIS_REVIEWER_GUIDE.md`, 236 lines, and `CONTRIBUTING.md`, which is actually a release-process doc) are misleadingly named. `CONTRIBUTING.md` does not actually tell anyone how to contribute.

### W5. The "convenience methods for pro users" pattern is sparse

The user mentioned that submodules should expose simple high-level commands that wrap complex mathematics, while still allowing pro users to drop down. The current code mostly *does* the second (pro users can wire everything up by hand) but rarely the first. There are a few good examples:

- `Sobolev.from_heat_kernel_prior(prior_scale, order, scale, ...)` on the sphere — wraps a lot of mathematics into one line.
- `LinearForwardProblem.from_direct_sum([fp1, fp2, ...])` — wraps the joint-inversion setup.
- `GaussianMeasure.credible_set(probability, geometry="ellipsoid", ...)` — clean and powerful.

But for the common cases users will reach for first — "give me a smooth prior on the unit interval", "give me a noise model with known variance and given sensor layout", "solve this Tikhonov problem with the standard sensible defaults", "give me a prior ball of radius R around zero" — there are no one-line entry points. A submodule should be the place where these live, *one per cell of the §2 matrix*, but the pattern is not modelled clearly enough yet.

### W6. The Lowering Execution mission is a known unknown

The current branch (`mission/20260403-lowering-master`) is mid-flight. Phases 1–5 are complete (per git log). This work changes operator evaluation semantics: a `LinearOperator` will become a computation graph that can be *planned*, *fused*, and *materialised*. The mission is explicitly isolated in a planner layer, but Phase 6 (decide whether to merge) is unresolved.

**Risk:** if pygeoinf is promoted publicly *now*, then a major-version bump in 2026 that exposes lowering planner semantics will be disruptive. **Opportunity:** if positioned right, "we are about to ship a discretization-aware execution planner" is a *huge* differentiator. Either way, this needs an explicit call.

### W7. Dependency footprint is heavy for what's promised

`pyproject.toml` requires Python ≥ 3.12, `numpy ≥ 2.0`, `scipy`, `matplotlib`, `pyqt6` (a GUI toolkit!), `joblib`, `threadpoolctl`, `numba`, `ipympl`. Optional extras add `pyshtools`, `Cartopy`, `osqp`, `clarabel`, `plotly`. The required deps include `pyqt6` and `numba` which is a sticky install on some systems and adds substantial weight for a library whose pitch is "abstract mathematical inference". A user who just wants the abstract `HilbertSpace`+`LinearOperator` machinery is paying a 100+ MB tax.

**Fix:** move `pyqt6`, `ipympl`, possibly `numba` into optional extras (`[interactive]`, `[plot]`). Keep the core install lean (numpy + scipy + joblib).

### W8. The boundary between core and concrete is fuzzy

Items in the core package that arguably belong in submodules:

- `datasets.py` (USGS earthquakes, GSN stations) — these are application-specific.
- `data_assimilation/pendulum/` — a demo.
- `plot.py` (cartopy-dependent slices) — symmetric-space-specific plotting code is mixed with generic distribution plotting.
- `dynamical_system.py` (111 lines, unclear status).
- `symmetric_space/` — debatable, but probably should be a separate package.

This pollution makes the core look bigger than it is and ties pygeoinf to dependencies it should not own.

### W9. `Vector` is `TypeVar("Vector")` without `bound=`

Look at [hilbert_space.py:48](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/hilbert_space.py#L48):

```python
Vector = TypeVar("Vector")
```

No bound. Type checkers can't verify anything about vectors. In practice this is okay because operations go through `space.add` / `space.multiply`, but it limits the value of static typing for downstream users. A `Vector = TypeVar("Vector", bound=Any)` with explicit `Protocol` for the minimum operations (`__add__`, `__sub__`, `__mul__`, `copy`) would tighten the contract.

### W10. Some commit conventions and process artefacts leak into git history

The CLAUDE.md describes a strict commit convention (`Plan:`, `Phase:`, `Related:` trailers) — that's great for agent-driven dev. But it ties the repo's git history to a particular workflow. A potential community contributor who opens a PR with `fix: typo` will trip the convention. Worth either documenting this clearly in `CONTRIBUTING.md` (which currently exists but is a release doc, not a contribution doc) or relaxing the convention for outside contributors.

---

## 7. Opportunities (SWOT)

### O1. Establish the "submodule ecosystem" as a first-class concept

Concretely:

- Coin a name: `pygeoinf-X` (PyPI naming convention for submodules).
- Create an organisation on GitHub (`pygeoinf/`) that hosts:
  - `pygeoinf` (core)
  - `pygeoinf-symmetric` (line / circle / sphere / torus / plane)
  - `pygeoinf-intervalinf` (or rename `intervalinf` → `pygeoinf-interval`)
  - `pygeoinf-3d` (rename `pygeoinf3D`)
  - `pygeoinf-cookbook` (worked geophysical examples)
  - `pygeoinf-template` (the empty submodule template repo)
- Publish a tested compliance pack (`pip install pygeoinf-testing`) that any submodule author can run.

This is roughly the model used by `scikit-learn-contrib`, `xarray-contrib`, and `pint`. It scales socially (it's clear who is responsible for what) and technically (each submodule has its own release cadence and CI).

### O2. Position against an unmet community need

The geophysics, applied math, and Bayesian inference communities have several adjacent libraries:

| Library | Strength | Gap pygeoinf can fill |
|---------|----------|----------------------|
| `hIPPYlib` | PDE-constrained Bayesian inversion via FEniCS | Python-native, no FEniCS lock-in, discretization-pluggable |
| `MUQ` | C++ MCMC + sampling | Pure Python; Hilbert-space-first; convex/deterministic UQ |
| `cuqipy` | Python Bayesian, structured priors | Operator algebra; Backus-Gilbert; modular submodule pattern |
| `pyMC`, `numpyro` | Probabilistic programming | Function-space priors that aren't just GPs on grids |
| `scipy.optimize` | General optimization | Convex analysis with support functions; constrained inversion in function spaces |

The clearest **wedge** is: *"pygeoinf is what you reach for when your unknown is a function on a domain you control the discretization of, you want Bayesian and convex-analytic guarantees, and you want to ship code that survives a change of grid."* That's a sharp positioning, and nothing today owns that wedge in pure Python.

### O3. Tie the brand to a flagship application

The `sphere_dli_example` work-script ([pygeoinf/work/sphere_dli_example.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/work/sphere_dli_example.py)) is exactly the kind of end-to-end demo that should be polished into a JOSS paper or a conference example. It uses real IRIS GSN stations and USGS earthquakes, builds a phase-velocity-perturbation field on the sphere, infers spherical-cap averages, and reports admissible-region bounds. That's a "look at this" example that lands with geophysicists. It also exercises ~80% of the library surface, so promoting it forces the library to be polished.

### O4. Lean into "execution planning" as a strategic differentiator

The Lowering Execution mission opens the door to:

- Auto-fusing chains like `A.adjoint @ R^-1 @ A` into a single optimised kernel.
- Materialising matrices lazily and reusing them across different solver calls.
- Detecting when a `MassWeightedHilbertSpace` operator chain can be re-expressed on the underlying space and avoiding redundant mass multiplies.
- Optionally compiling hot paths via `numba` (already a dep).
- Future: lowering to GPU via JAX or PyTorch back-ends.

If shipped with a clear story, this gives pygeoinf a *performance* narrative that competitors lack. Most function-space libraries are slow because they implement everything via dense matrices; pygeoinf could become "fast and abstract" simultaneously.

### O5. Pedagogical leverage

The mathematical clarity of pygeoinf makes it a natural teaching tool. There is an opportunity to develop:

- A short course or textbook chapter ("Bayesian Inverse Problems in Function Spaces, in Python") tied to the library.
- A "Notebook of the Week" series on the sphere, interval, and 3D box submodules.
- A teaching companion repo with exercises (e.g. "implement a `Hermite` `HilbertSpace` and verify the axioms").

This pulls in students, postdocs, and faculty — exactly the kind of organic user base that produces long-tail contributors.

### O6. JOSS / SIAM-J-SISC paper

The library has enough novel content (axiom-checked Hilbert spaces, operator dual/adjoint distinction, dual-master Backus-Gilbert, function-space Gaussian credible sets including the weakened-ellipsoid construction documented in [convex-analysis-reference.md:93-99](/home/adrian/PhD/Inferences/pygeoinf/docs/agent-docs/references/living/convex-analysis-reference.md#L93-L99)) for one or two papers. JOSS is the fastest path to citation; SIAM Journal on Scientific Computing or Geophysical Journal International is the heaviest. Probably both — JOSS first to anchor citations.

### O7. Community contribution patterns

The axiom-check mixin pattern is *itself* publishable as a methodology. A blog post or short paper titled "Axiom-checked abstract base classes for scientific Python" would draw attention to the pattern and to pygeoinf as its exemplar.

---

## 8. Risks (SWOT)

### R1. A bad first release fragments the user base

If pygeoinf is promoted before the submodule contract is stable, early adopters will write submodules to the current implicit contract, then break when the contract is formalised. **Mitigation:** publish v2.0 with the contract baked in *before* doing any wider promotion.

### R2. The Lowering mission may force breaking changes

If `LinearOperator` semantics change as part of Phase 6, every downstream submodule and every user script breaks. **Mitigation:** decide *now* whether the mission lands in pygeoinf 2.0 or stays as a planner layer; advertise the decision in the manifesto.

### R3. Heavy dependencies will scare casual users

`pyqt6` as a hard dep is a particular landmine on macOS and headless servers. **Mitigation:** trim deps to numpy/scipy/joblib in `pyproject.toml`, with everything else optional, before any promotion.

### R4. The agentic-doc style will alienate human contributors

`CLAUDE.md`, `AGENTS.md`, `.github/agents/*.agent.md`, agent-docs all signal "this is an AI-collaborator-first repo". That's fine *if* it is paired with strong human-facing docs. Without that, outside contributors will conclude they aren't welcome. **Mitigation:** add a high-quality `CONTRIBUTING.md` aimed at humans; add a CODE_OF_CONDUCT; have a `HUMANS_VS_AGENTS.md` that explains the duality openly.

### R5. The `intervalinf` / `pygeoinf3D` setup will confuse newcomers

A user lands on the GitHub org, sees four sibling repos, and has no idea which to install. Today, the architecture means they need pygeoinf core + one or more specialised packages. There is no `pip install pygeoinf` that "just works" for a 1D problem. **Mitigation:** the meta-package pattern — `pip install pygeoinf[interval,sphere]` could pull the right siblings via extras.

### R6. Convex-analysis correctness is hard to audit

Some of the most powerful and most novel code lives in `convex_analysis.py`, `convex_optimisation.py`, `backus_gilbert.py`, `gaussian_measure.py::credible_set`. The math is genuinely subtle (subgradient correctness, weakened-ellipsoid trace calibration, weighted chi-square quantiles via Imhof/saddlepoint). A latent bug here is exactly the kind that erodes scientific trust if a high-profile user catches it. **Mitigation:** publish a *reviewer guide* (one exists as a 236-line top-level file but is not in the polished docs), and consider an external math review (or formal verification of key identities) before promotion.

### R7. Maintenance burden grows superlinearly with submodules

Once the submodule ecosystem exists, breaking changes in pygeoinf core impose cost on every submodule. **Mitigation:** semver discipline, an LTS branch policy, and the compliance-test pack so any breaking change is detected by submodule CI before it reaches users.

### R8. Test runtime will balloon

233 tests today, ~14k lines of test code. CI matrix is 3 OS × 1 Python = 3 jobs. Adding more concrete spaces and properties means more cross-product tests. **Mitigation:** invest in a tiered test layout (fast unit / slow integration / numerical-heavy) and run only fast tests on every PR.

### R9. Brand confusion

`pygeoinf` (geophysical inference) vs `pygeoinf3D` vs `intervalinf` vs `seismic_tomo`. The names tell a contradictory story (some "geo" prefixed, some not; some "inf" suffixed, some not). **Mitigation:** rename or alias under a consistent scheme before public promotion. The natural scheme is `pygeoinf` + `pygeoinf-<scope>`.

### R10. The "discretization-agnostic" promise is partially structural-only

`HilbertSpace.dim` is required to be a finite int. So pygeoinf is honest: it is "discretization-agnostic about which discretization you choose, but you still have to discretize". This nuance gets lost in marketing. **Mitigation:** be precise in the manifesto. "Pygeoinf abstracts the *choice of discretization*, not the *act of discretization*."

---

## 9. Detailed module-by-module audit

This section is the small-scale view. For each module, I list its current role, line count, principal abstractions, notable strengths, and any flags worth attention.

### 9.1 Hilbert layer

#### `hilbert_space.py` — 850 lines

- **Role.** Foundational. Defines `HilbertSpace`, `DualHilbertSpace`, `HilbertModule`, `EuclideanSpace`, `MassWeightedHilbertSpace`, `MassWeightedHilbertModule`.
- **Strengths.** Riesz-map-defined inner product; `from_formal_adjoint` for mass-weighted lifts; `coordinate_inclusion`/`coordinate_projection`/`riesz`/`inverse_riesz` as `LinearOperator`-valued properties.
- **Flags.** `Vector = TypeVar("Vector")` lacks `bound=`. `dim` is required to be a finite int (cylinders/Banach not supported). `is_element` default uses `isinstance(x, type(self.zero))`, which is brittle for some representations and worth flagging in the docs.

#### `linear_forms.py` — 263 lines

- **Role.** Concrete `LinearForm` representation of dual elements via component vector.
- **Strengths.** In-place arithmetic (`__iadd__`, `__imul__`); auto-component computation from mapping with optional `joblib` parallelism; `from_linear_operator` factory for rank-1 ops.
- **Flags.** None significant.

#### `nonlinear_forms.py` — 340 lines

- **Role.** `NonLinearForm` ABC supporting gradient, Hessian, subgradient oracles.
- **Strengths.** Clean separation of differentiable vs subdifferentiable; supports the convex-analysis stack downstream.
- **Flags.** None significant.

#### `direct_sum.py` — 534 lines

- **Role.** `HilbertSpaceDirectSum`, `BlockLinearOperator`, `ColumnLinearOperator`, `RowLinearOperator`, `BlockDiagonalLinearOperator`.
- **Strengths.** Vectors are *lists* of vectors from subspaces — clean, recursive, supports arbitrary depth of nesting. Used by `LinearForwardProblem.from_direct_sum`, `GaussianMeasure.from_direct_sum`, `joint_measure`.
- **Flags.** Mixed Galerkin / standard matrix handling is subtle (good unit tests exist).

### 9.2 Operator algebra

#### `linear_operators.py` — 1602 lines

- **Role.** `LinearOperator` + factory methods (`from_matrix`, `from_linear_forms`, `from_vectors`, `from_tensor_product`, `from_formal_adjoint`, `self_dual`, `self_adjoint`, …), specialised subclasses (`MatrixLinearOperator`, `DenseMatrixLinearOperator`, `SparseMatrixLinearOperator`, `DiagonalSparseMatrixLinearOperator`).
- **Strengths.** Rich operator algebra (`@`, `+`, `*`, `-`, `__neg__`), automatic adjoint/dual deduction, parallel dense-matrix assembly, diagonal-extraction utilities (memory-efficient for preconditioners).
- **Flags.** The default "compute adjoint via dual via mapping" fallback is correct but slow ([line 99-115](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/linear_operators.py#L99-L115)); user docs should warn that low-dimensional fallback is for testing only.

#### `affine_operators.py` — 174 lines

- **Role.** `AffineOperator(linear_part, translation)`, returned by `kalman_operator` etc. when expectations are non-zero.
- **Strengths.** Coherent algebra with `LinearOperator` via `__add__`/`__sub__` return-NotImplemented dispatch.
- **Flags.** Small and focused.

#### `nonlinear_operators.py` — 219 lines

- **Role.** `NonLinearOperator` base. `LinearOperator` inherits.
- **Strengths.** Clean derivative oracle interface.
- **Flags.** None significant.

#### `spectral_operator.py` — 261 lines

- **Role.** `SpectralFractionalOperator` for `f(A)` via spectral diagonalisation; powers the weakened-ellipsoid credible-set construction.
- **Flags.** Worth merging into a future `pygeoinf.matrix_function` subpackage with `matrix_function.py`.

#### `matrix_function.py` — 216 lines

- **Role.** `apply_matrix_function` via Lanczos/Krylov for `f(A) v`.
- **Flags.** None significant; could absorb `spectral_operator.py`.

### 9.3 Probability

#### `gaussian_measure.py` — 2075 lines

- **Role.** The library's centre of gravity for probability.
- **Strengths.** Operator-valued covariance; multiple constructors; affine pushforward (`affine_mapping`); credible-set machinery; direct-sum measures; KL divergence; sampling via Cholesky / eigendecomposition / SVD.
- **Flags.** **Largest beneficiary of a split.** Suggested decomposition:
  - `pygeoinf.gaussian_measure.core` — class definition, basic properties.
  - `pygeoinf.gaussian_measure.constructors` — `from_standard_deviation`, `from_samples`, `from_factor`, …
  - `pygeoinf.gaussian_measure.credible_sets` — `credible_set`, `ambient_ball`, `weakened_ellipsoid`.
  - `pygeoinf.gaussian_measure.low_rank` — low-rank approximations.
  - `pygeoinf.gaussian_measure.algebra` — `__add__`, `__sub__`, `affine_mapping`, `from_direct_sum`.

#### `quadratic_form_quantile.py` — 526 lines

- **Role.** Weighted-chi-square quantiles via Imhof / saddlepoint / Monte Carlo, used by `credible_set`.
- **Strengths.** Auto-selection heuristic between backends.
- **Flags.** Likely should move into the `gaussian_measure` subpackage as `credible_sets._quantiles`.

### 9.4 Sets, subspaces, convex analysis

#### `subspaces.py` — 799 lines

- **Role.** `LinearSubspace`, `AffineSubspace`, `OrthogonalProjector`.
- **Strengths.** Standard but solid.
- **Flags.** None.

#### `subsets.py` — 1713 lines

- **Role.** Subset hierarchy: `Subset`, `EmptySet`, `UniversalSet`, `Complement`, `Intersection`, `Union`, `SublevelSet`, `LevelSet`, `ConvexSubset`, `Ellipsoid`, `Ball`, `Sphere`, …
- **Strengths.** CSG-style combinators; convex-intersection specialisation via max-functional combination.
- **Flags.** 1713 lines is plenty. The plot-related methods on `Subset` could be split out.

#### `convex_analysis.py` — 923 lines

- **Role.** Support functions of closed convex sets.
- **Strengths.** Polished; well-tested ([tests/test_support_function_constructors.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_constructors.py), [tests/test_support_function_algebra.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_algebra.py)).
- **Flags.** None significant.

#### `convex_optimisation.py` — 2601 lines

- **Role.** Three algorithms: `SubgradientDescent`, `ProximalBundleMethod`, `PrimalKKTSolver`.
- **Strengths.** Algorithms are deeply tied to convex-analysis primitives, which is conceptually clean.
- **Flags.** **Split into subpackage.** Suggested decomposition:
  - `pygeoinf.convex_optimisation.subgradient` — `SubgradientDescent`, `SubgradientResult`.
  - `pygeoinf.convex_optimisation.bundle` — `ProximalBundleMethod`, `BundleResult`, level-bundle variants.
  - `pygeoinf.convex_optimisation.primal_kkt` — `PrimalKKTSolver`, `KKTResult`.
  - `pygeoinf.convex_optimisation.qp_backends` — wrappers for OSQP, Clarabel, etc.

### 9.5 Solvers and optimisation

#### `linear_solvers.py` — 1053 lines

- **Role.** `LinearSolver` ABC + direct (`LU`, `Cholesky`, `Eigen`) and iterative (`CG`, `MinRes`, `BICGStab`, `FCG`, plus matrix-wrapped Scipy iterative solvers).
- **Strengths.** Pure matrix-free abstract solvers exist alongside SciPy-backed solvers.
- **Flags.** Could be split into `linear_solvers.{direct,iterative,callbacks}`.

#### `preconditioners.py` — 549 lines

- **Role.** `JacobiPreconditioningMethod`, `SpectralPreconditioningMethod`, `IterativePreconditioningMethod`, `BandedPreconditioningMethod`, `ExactBlockPreconditioningMethod`, `ColumnThresholdedPreconditioningMethod`.
- **Strengths.** Each preconditioner returns a `LinearOperator`, plays into the solver API.
- **Flags.** Significant under-exposure: users won't know which preconditioner to pick. Needs a "preconditioner cookbook" doc page.

#### `low_rank.py` — 986 lines

- **Role.** `LowRankSVD`, `LowRankEig`, `LowRankCholesky`, randomized algorithms, `white_noise_measure`.
- **Strengths.** All algorithms are matrix-free in the abstract sense.
- **Flags.** None significant.

#### `linear_optimisation.py` — 1330 lines

- **Role.** `LinearLeastSquaresInversion`, `LinearMinimumNormInversion`, constrained variants.
- **Strengths.** Same `formalism="model_space"|"data_space"` knob as Bayesian; minimum-norm via discrepancy principle with analytical Fréchet derivatives.
- **Flags.** None significant.

#### `linear_bayesian.py` — 1154 lines

- **Role.** `LinearBayesianInversion`, plus various preconditioner factories.
- **Strengths.** Randomize-then-optimise sampling for the posterior; dual formalism; specialized preconditioners.
- **Flags.** None significant; if anything, the most polished of the inversion modules.

#### `nonlinear_optimisation.py` — 218 lines

- **Role.** `ScipyUnconstrainedOptimiser` wraps `scipy.optimize.minimize`.
- **Strengths.** Thin and pragmatic.
- **Flags.** Could be a starting point for a richer nonlinear-inversion story.

### 9.6 Inverse-problem orchestration

#### `forward_problem.py` — 432 lines

- **Role.** `ForwardProblem`, `LinearForwardProblem`.
- **Strengths.** Clean factoring; chi-squared tests in-built; `parameterized_problem` / `data_reduced_problem` for SOLA-style problem reductions.
- **Flags.** None significant.

#### `inversion.py` — 312 lines

- **Role.** `Inversion`, `LinearInversion` (formalism), `LinearInference`.
- **Flags.** Could absorb the inversion-side joint methods now scattered across `linear_bayesian.py` and `linear_optimisation.py`.

#### `backus_gilbert.py` — 399 lines

- **Role.** `DualMasterCostFunction`.
- **Strengths.** Differentiated capability; well-tested via the `sphere_dli_example`.
- **Flags.** A legacy `BackusInference` is referenced in the living reference but may have been removed or renamed — worth a re-check.

### 9.7 Concrete spaces

#### `symmetric_space/` — 6 files

- **Role.** `SymmetricHilbertSpace` ABC + `AbstractSymmetricLebesgueSpace` + `SymmetricSobolevSpace`; concretes for `circle`, `line`, `plane`, `sphere`, `torus`.
- **Strengths.** Exemplary modular pattern. `InvariantLinearAutomorphism` + `InvariantGaussianMeasure` use `DiagonalSparseMatrixLinearOperator` for O(n) algebra. `sphere.Lebesgue` uses `pyshtools` for spherical harmonic transforms; cap-quadrature for radial integrals.
- **Flags.** Lives inside pygeoinf — should be carved out (see §10). Optional deps (`pyshtools`, `Cartopy`) make this a borderline case for in-tree.

### 9.8 Auxiliary

#### `datasets.py` — 226 lines

- **Role.** GSN station catalogue, USGS earthquake fetcher, sampling utilities.
- **Flags.** Application-specific. Should move to a submodule or to `pygeoinf-cookbook`.

#### `plot.py` — 2098 lines

- **Role.** `plot_1d_distributions`, `plot_corner_distributions`, `SubspaceSlicePlotter`, `plot_slice`, plus map plotters.
- **Flags.** **Second-largest beneficiary of a split.** Suggested:
  - `pygeoinf.plot.distributions` — 1D, corner plots.
  - `pygeoinf.plot.subset_slice` — `plot_slice`, `SubspaceSlicePlotter`.
  - `pygeoinf.plot.maps` — cartopy-dependent code → move into `pygeoinf-symmetric` plug-in.

#### `parallel.py` — 73 lines

- **Role.** `joblib`-backed parallel-matrix utilities.
- **Flags.** None.

#### `data_assimilation/` (pendulum demo)

- **Role.** Tutorial-grade demo.
- **Flags.** Move to `pygeoinf-cookbook`.

#### `dynamical_system.py` — 111 lines

- **Role.** Unclear (small file). Worth a quick decision: keep, document, or move.

### 9.9 Axiom checks (`checks/`)

- **Role.** Mixin classes for `HilbertSpace`, `LinearOperator`, `NonLinearOperator`, `AffineOperator`.
- **Strengths.** Discoverable, randomised, exhaustive on the axioms it tests.
- **Flags.** **Underexposed.** Should be promoted to `pygeoinf.testing` for explicit use by submodule authors.

---

## 10. Roadmap proposal

I propose three horizons. Each step has a *concrete artefact* and a *measurable success criterion*. **Horizon 1 is dominated by delivering the 2×2 matrix from §2** — everything else in H1 is in service of that.

### 10.1 Horizon 1 — "Ship the 2×2 matrix and harden for v2.0" (3 months)

Goal: get to a v2.0 release that is *opinionated about workflows*, *symmetric across paths and modes*, and that we are willing to point researchers at without caveats.

The first nine steps (H1.A–H1.I) deliver the matrix. They are also the deepest design work in the whole roadmap; the rest (H1.1–H1.7) is reorganisation around the new shape.

| Step | Artefact | Done when |
|------|----------|-----------|
| **H1.A** | **`Likelihood` and `SetLikelihood` classes.** A `GaussianLikelihood(forward_operator, noise_measure)` and `SetLikelihood(forward_operator, error_set)`. Both expose `__call__(model) → measure_on_D` / `→ set_in_D`, `log_density(model, data)` / `contains(model, data)`, and `as_data_distribution_given_model(model)`. | Two classes, public API, with unit tests. `LinearForwardProblem` is refactored to take either a measure or a set (or expose a `.likelihood` property that returns the appropriate object). |
| **H1.B** | **`LinearInference` parent class.** Mirror of `LinearInversion`, holds (forward_problem, property_operator, property_space) plus the formalism switch. | Class lives in `inversion.py`; `LinearInversion` and `LinearInference` share a common ABC. |
| **H1.C** | **`LinearBayesianInference` class.** Cell (measure, inference). Computes the Gaussian posterior on P = T(M) by pushing forward `LinearBayesianInversion`'s posterior. Supports the dual `formalism` switch. Convenience: `property_posterior_measure`, `expectation_operator`, `kalman_operator`, sampler. | Class with full docstring + 6+ tests; appears in `__init__.py`. |
| **H1.D** | **`LinearSetInversion` class.** Cell (set, inversion). Takes (forward_problem, model_prior_set, data_error_set, data) and returns the admissible region in M (as a `Subset` or as a `SupportFunction`). Picks the appropriate convex solver internally. | Class with full docstring + tests; existing `Constrained{LeastSquares,MinimumNorm}` classes are *re-presented* as special cases of this in the docs. |
| **H1.E** | **`LinearSetInference` class.** Cell (set, inference). Wraps the `DualMasterCostFunction` per-direction loop into a single user-facing class. The `sphere_dli_example.py` work-script reduces to ~30 lines using this class. | Class lives in `backus_gilbert.py` (or moved to `inversion.py`); the sphere DLI example is rewritten on top of it and still passes its 13 tests. |
| **H1.F** | **Path bridges are named.** Add a `pygeoinf.bridges` module (or use the existing `GaussianMeasure.credible_set` with deliberate documentation). Implement at least one *softening* bridge (`Subset.gaussian_envelope(...)` for ball and ellipsoid). | `pygeoinf.bridges.harden(measure, probability, geometry=…)` and `pygeoinf.bridges.soften(subset, probability=…)` exist as the documented public API; backed by existing `credible_set` and new softening implementations. |
| **H1.G** | **Mode bridges are named.** `InversionResult.infer(property_operator, property_space)` and `InferenceResult.lift_to_inversion(...)` (the latter as a documented best-effort, with an explicit warning that this is generally ill-posed). | Methods exist on both result types; tested. |
| **H1.H** | **Unified result types.** `InversionResult` carries (posterior_measure OR admissible_region, diagnostics). `InferenceResult` carries the property-space analogue. Both expose `.harden(...)`/`.soften(...)`/`.infer(...)` per the bridge framework. | Both result types defined; the four workflow classes return them. |
| **H1.I** | **Four "fast-path" tutorials**, one per cell of the matrix. Each is ≤ 30 lines of user code. | Tutorials 1a, 1b, 1c, 1d (or named tutorials) are published; each runs on Colab in < 5 min. |
| H1.1 | **Manifesto.** A 1–2 page document, on the front page of docs and as `MANIFESTO.md` in repo, that pins the four claims (mathematics-first, discretization-agnostic, modular orchestrator, linear-focused for now), shows the 2×2 matrix from §2, and links to the four fast-path tutorials. | Reviewed and signed off by the three core devs. |
| H1.2 | **Public API contract.** Split `__init__.py` into `core`, `advanced`, `experimental`. Document semver semantics. | Each tier has its own subsection in the API docs, with a stability badge. |
| H1.3 | **Submodule contract.** Publish `pygeoinf.testing` (re-export of axiom-check mixins as a standalone compliance pack). | A new minimal example submodule (`pygeoinf-toyspace`) can be built in <100 lines and passes `pygeoinf.testing.assert_complete_submodule`. |
| H1.4 | **Slim core install.** Move `pyqt6`, `ipympl`, possibly `numba`, into optional extras. | `pip install pygeoinf` installs in <30 s on a clean venv. |
| H1.5 | **Refactor the three mega-modules** (`gaussian_measure`, `convex_optimisation`, `plot`) into subpackages. | All existing tests still pass; no public-API regression. |
| H1.6 | **Resolve Lowering Mission Phase 6.** Decide: merge planner into core (v2.x major), keep as planner-only (v1.x minor), or shelve. | Decision documented in the manifesto. |
| H1.7 | **Polish `CONTRIBUTING.md`** for humans (the current one is a release-process doc). Add CODE_OF_CONDUCT.md and the human-vs-agent collaboration note. | An external first-time contributor can open a PR following the docs only. |

### 10.2 Horizon 2 — "Ecosystem launch" (6 months)

Goal: have a working ecosystem of named submodules and a flagship application built on the 2×2 matrix.

| Step | Artefact | Done when |
|------|----------|-----------|
| H2.1 | **Carve out `pygeoinf-symmetric`.** `symmetric_space/` becomes its own repo and PyPI package. Pin it as the canonical example submodule in the manifesto. Add convenience methods per workflow (e.g. `Sobolev.heat_kernel_prior`, `Sobolev.ball_prior_set`). | `pip install pygeoinf-symmetric` works; old import paths still work via a thin shim that emits a `DeprecationWarning`. |
| H2.2 | **Promote `intervalinf` → `pygeoinf-interval`** and `pygeoinf3D` → `pygeoinf-3d`. Each submodule exposes natural one-line constructors for the priors most useful in its domain. | Same as H2.1; docs cross-link; each submodule's tutorial covers at least one cell of the 2×2 matrix. |
| H2.3 | **`pygeoinf-cookbook` repo.** One real end-to-end inversion *per cell of the 2×2 matrix*, on each of the three sibling submodules. Notebooks must run in CI. | At least 4×3 = 12 cookbook examples; each ≤ 200 lines of code. |
| H2.4 | **Flagship application: `sphere_dli_example` paper.** JOSS submission + at least one geophysics conference talk. The paper explicitly frames it as a "set-inference" workflow exemplar. | Submitted. |
| H2.5 | **GitHub org `pygeoinf/`** consolidates all repos. Discussions enabled. Issue templates standardised. | All sibling repos live under the org. |
| H2.6 | **Preconditioner cookbook.** A single doc page that walks through choosing among the six preconditioners. | Page lives in `docs/source/`. |
| H2.7 | **JAX/Torch back-end exploration.** A spike that wires a `JAXLinearOperator` and runs an example through it. Outcome is yes/no on integrating with the autodiff ecosystem. | Spike report in `docs/agent-docs/` (no commitment to ship). |

### 10.3 Horizon 3 — "Community and growth" (12 months)

Goal: pygeoinf has a small but active community of submodule authors and a clear story for the next two years.

| Step | Artefact | Done when |
|------|----------|-----------|
| H3.1 | **At least one third-party submodule.** A researcher outside the core team has published a `pygeoinf-X` package. | Listed in the official ecosystem page. |
| H3.2 | **Citations.** ≥ 5 citations to the JOSS paper. | Measured at month 12. |
| H3.3 | **Teaching companion.** A short-course or graduate-class set of notebooks built on pygeoinf, organised around the 2×2 matrix. | Adopted by at least one external instructor. |
| H3.4 | **Performance story.** A benchmark suite (operations-per-second on the sphere, on the interval, on a 3D box) published in the docs. | Updated automatically by CI on each release. |
| H3.5 | **Long-term support (LTS) policy.** Document which version is LTS and what the deprecation window is. | Published in `MANIFESTO.md` v2. |
| H3.6 | **Annual community survey.** Ask the 50–100 users what they want next. | Run once. |
| H3.7 | **Decision: extend to non-linear?** With the linear story solid, decide whether to extend the 2×2 matrix to non-linear forward operators (with linearisation-based or sample-based variants). | Decision documented; if yes, a Horizon 4 plan exists. |

### 10.4 What *not* to do

- Don't try to be all things to all geophysicists. Don't add MCMC samplers, don't add a deep-learning backend, don't add seismic-specific physics modules in core. Those belong in submodules.
- Don't break the abstract Hilbert space contract for performance — performance lives in the planner / lowering layer, not in core abstractions.
- Don't promote publicly before H1 is done. Stability is currency.

---

## 11. Open questions for the dev team

These are decisions the report cannot make on the team's behalf.

1. **Naming of the workflow classes (§2).** Are `LinearBayesianInversion` / `LinearBayesianInference` / `LinearSetInversion` / `LinearSetInference` the right names? Alternatives: `Linear{Probabilistic,Convex}{Inversion,Inference}`, or keep the existing Bayesian terminology paired with `LinearAdmissible{Inversion,Inference}` for the set path. The choice affects every tutorial and every docstring. Locking the names is the prerequisite to H1.A–H1.I.
2. **`LinearForwardProblem` refactor.** Should the data-error attribute become a `Union[GaussianMeasure, Subset]`, or should there be sibling classes `LinearGaussianForwardProblem` and `LinearSetForwardProblem`? My recommendation: union, because it keeps the path-bridge story symmetric, but the team has more context on downstream implications.
3. **Granularity of inference.** Is *inference* always linear (the property is `T(u)` for some linear T) or sometimes non-linear (e.g. `||u||`, eigenvalues of some operator built from u)? Pygeoinf today is linear-property-only. Locking this affects the `Property{Space,Operator}` abstractions.
4. **Is the Lowering Execution Framework going to land in pygeoinf core, or stay as an external planner?** Without a decision here, the v2.0 major-version cut can't be planned.
5. **What is the policy on agentic-collaborator-flavoured artefacts (`CLAUDE.md`, `AGENTS.md`, `.github/agents/`)?** Will they remain in-tree and visible, or move to a separate `dev-tools` repo? My recommendation: keep them, but pair with strong human-facing docs that openly acknowledge the dual mode.
6. **Renaming.** Is the team willing to rename `pygeoinf3D` → `pygeoinf-3d` and `intervalinf` → `pygeoinf-interval`? Or is the existing naming entrenched enough that the cost of rename outweighs the benefit?
7. **Versioning.** Is the team comfortable bumping straight to v2.0 for the manifesto + 2×2 matrix release? My recommendation: yes — the symbolic value of "we drew a line and we promised four named workflows" is significant.
8. **Funding / time.** What is the realistic dev capacity (FTE-equivalent) over the next 3 / 6 / 12 months? Horizon 1 is heavier than before; trimming H2 or H3 is fine if capacity is lower, but H1.A–H1.I are non-negotiable for the v2.0 story.
9. **Target user.** Concretely, who is *user persona #1*? My read is "a PhD-level applied mathematician or geophysicist who wants to choose between Bayesian and set-theoretic linear inverse problems in a function space without rolling their own linear algebra". If that's wrong, the docs and tutorials need re-aiming.
10. **Submodule registry.** Should there be a curated list, or an open ecosystem? Curated is safer initially but caps growth; open is risky early but scales.
11. **Convex analysis & credible sets — is this a co-equal pillar of the brand?** §2 takes the strong position that *yes*, the set path is co-equal with the measure path. This is differentiated but unfamiliar to most users. Confirm this is the team's view.
12. **Non-linear scope.** §10.3 has a milestone "decide whether to extend to non-linear". Is the team OK with non-linear being explicitly *out of scope* for the next 12 months?

---

## Appendix A — Cross-references

- README: [pygeoinf/README.md](/home/adrian/PhD/Inferences/pygeoinf/README.md)
- Public API: [pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py)
- Existing living references:
  - [backus-gilbert-reference.md](/home/adrian/PhD/Inferences/pygeoinf/docs/agent-docs/references/living/backus-gilbert-reference.md)
  - [convex-analysis-reference.md](/home/adrian/PhD/Inferences/pygeoinf/docs/agent-docs/references/living/convex-analysis-reference.md)
  - [sphere-dli-example-reference.md](/home/adrian/PhD/Inferences/pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md)
- Submodule exemplars:
  - In-tree: [pygeoinf/symmetric_space/](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/symmetric_space)
  - External: [intervalinf/](/home/adrian/PhD/Inferences/intervalinf), [pygeoinf3D/](/home/adrian/PhD/Inferences/pygeoinf3D)
- CI: [.github/workflows/ci.yml](/home/adrian/PhD/Inferences/pygeoinf/.github/workflows/ci.yml)
- Active plans: [docs/agent-docs/active-plans/](/home/adrian/PhD/Inferences/pygeoinf/docs/agent-docs/active-plans/)
- Tutorials: [tutorials/](/home/adrian/PhD/Inferences/pygeoinf/tutorials/)
- Flagship work-script: [pygeoinf/work/sphere_dli_example.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/work/sphere_dli_example.py)

## Appendix B — Quick stats

- Python files in `pygeoinf/`: 33 core + 6 in `symmetric_space/` + 4 in `checks/` + small auxiliary.
- Total package LOC: ~22,300.
- Tests: 60 files, ~14,000 LOC, 233 `test_*` functions, 494 collected items per recent CI snapshot.
- Tutorials: 12 (`tutorial1.ipynb` … `tutorial10.ipynb` + 2 demos).
- Public symbols exported by `__init__.py`: 75.
- Largest five modules: `convex_optimisation` (2601), `plot` (2098), `gaussian_measure` (2075), `subsets` (1713), `linear_operators` (1602).
- Sphinx docs: scaffolded but thin (`index.rst` is essentially a re-rendered README).
- Agent-docs: thorough; ~13k lines across `active-plans/`, `completed-plans/`, `references/`, `theory/`.
- Python: `>=3.12` only.

End of report.
