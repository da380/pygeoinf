# DualMasterCostFunction — Faster / More Robust Convex Optimisation

**Scope:** Robust, practical algorithms to compute

$$h_U(q) = \inf_{\lambda \in D}\; \varphi(\lambda; q)$$

where `pygeoinf.backus_gilbert.DualMasterCostFunction` implements

$$\varphi(\lambda; q)
  = \langle \lambda, \tilde d \rangle_D
  + \sigma_B\!\bigl(T^*q - G^*\lambda\bigr)
  + \sigma_V(-\lambda).$$

**Motivation:** The current `pygeoinf.convex_optimisation.SubgradientDescent` uses a fixed step size and provides no optimality certificate. Dual master problems are convex but typically non-smooth; we want (i) faster convergence, (ii) robustness without manual step-size tuning, (iii) a reliable stopping criterion based on an optimality gap, and (iv) scalability to many directions $q \in P = \mathbb{R}^p$ (expected: tens of directions).

---

## 1. Current State in pygeoinf (What We Can Build On)

**Existing pieces:**
- `pygeoinf.backus_gilbert.DualMasterCostFunction`
  - Has `set_direction(q)` which caches $T^* q$.
  - Provides a **subgradient oracle** via support points: `sigma.support_point(z)`.
- `pygeoinf.convex_analysis.SupportFunction`
  - Base class for support functions with `support_point(q) -> Optional[Vector]`.
  - Concrete implementations: `BallSupportFunction`, `EllipsoidSupportFunction`, half-space support.
- `pygeoinf.convex_optimisation.SubgradientDescent`
  - Minimal constant-step implementation; acceptable as a baseline, not production-grade.
- `pygeoinf.parallel` — joblib-based helpers, usable for multi-direction parallelism.

**Observed limitations:**
- Constant-step subgradient is sensitive to scaling and step choice; convergence is $O(1/\sqrt{k})$ at best, and certificates are unavailable.
- The non-smoothness of the dual master is *structured* (support of ball = norm, support of ellipsoid = weighted norm); this structure is not exploited.

---

## 2. Bundle / Level Bundle Methods (Primary Recommendation)

### 2.1 Why bundle methods are the right fit

Bundle methods were developed precisely for the class of problems the dual master belongs to: convex non-smooth objectives presented via a value + subgradient oracle. The rationale is:

- They accumulate **cutting-plane underestimators** of $\varphi$, yielding a lower bound $f_\mathrm{low}$.
- The best evaluated value gives an upper bound $f_\mathrm{up}$.
- The **optimality gap** $\Delta_k = f_\mathrm{up} - f_\mathrm{low}$ is a reliable, tuning-free stopping criterion.
- A **stability center** $\hat\lambda$ prevents the oscillation that plagues plain subgradient methods.
- Convergence is $O(1/k)$ for the gap (vs $O(1/\sqrt{k})$ for subgradient with optimal step schedule); in practice bundles are dramatically faster on structured problems.

The `pygeoinf/plans/bundle-methods-optimizer-plan.md` already documents a detailed implementation plan for this. The key references are Lemaréchal (1974, 1978) for the original bundle method, and Lemaréchal, Nemirovskii & Nesterov (1995) and Kiwiel (1995, 1990) for level/proximal variants.

### 2.2 Cutting-plane model

Given a collection of oracle evaluations $(\lambda_j, f_j, g_j)$ with $g_j \in \partial \varphi(\lambda_j)$, the piecewise-linear cutting-plane model is

$$\hat{\varphi}_k(\lambda)
  = \max_{j \in J_k}\bigl\{ f_j + \langle g_j,\, \lambda - \lambda_j \rangle \bigr\}
  \;\le\; \varphi(\lambda).$$

The lower bound is $f_{\mathrm{low},k} = \min_\lambda \hat\varphi_k(\lambda)$.

### 2.3 Level bundle master problem

One iteration of the level bundle method solves a **small QP**: given the current stability center $\hat\lambda_k$ and level

$$f_{\mathrm{lev},k} = \alpha\, f_{\mathrm{low},k} + (1 - \alpha)\, f_{\mathrm{up},k},
\qquad \alpha \in (0, 1),$$

find

$$\lambda_{k+1} = \arg\min_\lambda \;\tfrac12 \|\lambda - \hat\lambda_k\|^2
\quad\text{s.t.}\quad \hat{\varphi}_k(\lambda) \le f_{\mathrm{lev},k}.$$

**Implementation-friendly epigraph form** (avoids the max in constraints):

$$\min_{\lambda,\, t}\; \tfrac12 \|\lambda - \hat\lambda_k\|^2
\quad\text{s.t.}\quad
  t \ge f_j + \langle g_j,\, \lambda - \lambda_j \rangle \;\forall j \in J_k,\qquad
  t \le f_{\mathrm{lev},k}.$$

This is a convex QP in $(\lambda, t)$ with $|J_k| + 1$ linear constraints. Bundle size $|J_k|$ is a tunable parameter (typical range: 50–200).

### 2.4 Serious and null steps

At each iteration:
- Evaluate $\varphi(\lambda_{k+1})$ and $g_{k+1} \in \partial \varphi(\lambda_{k+1})$.
- **Serious step** (sufficient descent): if $\varphi(\lambda_{k+1}) \le f_{\mathrm{up},k} - m \cdot \Delta_k$ for some $m \in (0,1)$, accept $\hat\lambda_{k+1} \leftarrow \lambda_{k+1}$ and update $f_{\mathrm{up}}$.
- **Null step**: otherwise, keep $\hat\lambda_{k+1} \leftarrow \hat\lambda_k$ and add the new cut.

Level infeasibility (can occur if $f_{\mathrm{lev},k}$ is below the cutting-plane minimum at $\hat\lambda$) is handled by re-centering or widening the level.

### 2.5 Proposed in-house implementation shape

Add to `pygeoinf/convex_optimisation.py`:

```python
@dataclass
class Cut:
    x: Vector          # oracle point
    f: float           # function value
    g: Vector          # subgradient
    iteration: int

class Bundle:
    """Manages a collection of cuts and tracks best (f_up, x_up)."""
    def add_cut(self, cut: Cut) -> None: ...
    def lower_bound(self) -> float: ...
    def linearization_constraints(self) -> tuple[np.ndarray, np.ndarray]: ...
    def compress(self, max_size: int) -> None: ...

@dataclass
class BundleResult:
    x_best: Vector
    f_best: float
    f_low: float
    gap: float
    converged: bool
    num_iterations: int
    num_serious_steps: int

class LevelBundleMethod:
    def __init__(self, alpha=0.1, tolerance=1e-6, max_iterations=500,
                 bundle_size=100, qp_solver=None): ...
    def solve(self, cost_fn, x0: Vector) -> BundleResult: ...

class ProximalBundleMethod:
    ...
```

### 2.6 QP backends

The master QP is small (typically $n_\lambda + 1$ variables, $|J_k| + 1$ constraints) but solved at every iteration. Backend choice is critical for performance.

**Always-available (SciPy-only):**
`scipy.optimize.minimize` with `method='SLSQP'`. Pros: no new dependencies. Cons: can be slow and numerically fragile for larger bundles; warm-starting is not supported.

**Recommended optional dependency — OSQP:**
[OSQP](https://osqp.org/) (Stellato et al. 2020) is a first-order ADMM-based QP solver that is specifically designed for repeated solves of problems with fixed structure. Key features relevant here:
- **Warm-starting**: OSQP explicitly supports warm-starting from a previous iterate; re-using the solver object across bundle iterations amortises the factorisation cost.
- Automatic adaptive $\rho$ parameter balances primal and dual residuals without manual tuning.
- Built-in infeasibility detection; handles the case where the level constraint makes the master QP infeasible gracefully.
- Install: `pip install osqp`.

**Alternative optional — Clarabel:**
[Clarabel](https://clarabel.org/) (Goulart & Chen, Oxford) is a modern interior-point solver with a **native QP path** that avoids epigraph reformulation of quadratic objectives, making it faster than HSDE-based solvers (SCS, ECOS) on QP instances. Also handles SOCPs and SDPs if the problem grows. Apache 2.0 licence.
- Install: `pip install clarabel`.

**Prototyping fallback — CVXPY:**
[CVXPY 1.8](https://www.cvxpy.org/) provides a high-level modelling interface and auto-dispatches to Clarabel/OSQP/SCS/HiGHS by default. Useful for rapidly prototyping the master QP's structure but too heavyweight for the inner loop of a production bundle method.

**Design recommendation:** plug backend via a small `Protocol`:

```python
class QPSolver(Protocol):
    def solve(self, P: np.ndarray, q: np.ndarray,
              A: np.ndarray, l: np.ndarray, u: np.ndarray,
              x0: np.ndarray | None = None) -> QPResult: ...
```

Backends: `SciPyQPSolver` (always), `OSQPQPSolver` (optional), `ClarabelQPSolver` (optional).

### 2.7 Fitting DualMasterCostFunction

The dual master oracle already computes $G^*\lambda$ and the residual $T^*q - G^*\lambda$ internally. Adding

```python
def value_and_subgradient(self, lam: Vector) -> tuple[float, Vector]:
```

allows bundle methods to get both $\varphi(\lambda)$ and $g \in \partial \varphi(\lambda)$ from one call, avoiding redundant operator applications. The subgradient formula is $g = \tilde d - Gv - w$ where $v = \sigma_B.\mathrm{support\_point}(\cdot)$ and $w = \sigma_V.\mathrm{support\_point}(\cdot)$.

---

## 3. Proximal Bundle Method (Secondary / More Robust Recommendation)

A **proximal bundle method** (Kiwiel 1990; Hiriart-Urruty & Lemaréchal 1993) replaces the level feasibility constraint with a proximal regularisation term. The master problem is:

$$\min_{\lambda,\, t}\; t + \tfrac{\rho_k}{2}\|\lambda - \hat\lambda_k\|^2
\quad\text{s.t.}\quad
  t \ge f_j + \langle g_j,\, \lambda - \lambda_j \rangle \;\forall j \in J_k.$$

**Advantages over level bundle:**
- The master QP is **always feasible** (no level infeasibility to handle).
- Simpler implementation; easier to get right with SciPy as the QP backend.
- Proximal term provides natural regularisation; the method is less sensitive to bundle size.

**Disadvantages:**
- Requires managing the **proximal weight** $\rho_k$: increase on null steps (tighter), decrease on serious steps (looser). A common rule is $\rho_{k+1} = \rho_k / \theta$ (serious) or $\rho_{k+1} = \rho_k \cdot \theta$ (null) for $\theta > 1$.
- The lower bound from the cutting-plane model is still available, so the gap certificate is preserved.

**Recommendation for implementation ordering:** implement proximal bundle before level bundle if SciPy is the only QP backend. The level bundle is the preferable long-term option once OSQP is available.

---

## 4. Primal–Dual / Saddle-Point Reformulation

### 4.1 Saddle reformulation

Support functions arise as suprema over their defining sets:

$$\sigma_B(z) = \sup_{m \in B}\langle z, m\rangle, \qquad
  \sigma_V(w) = \sup_{v \in V}\langle w, v\rangle.$$

Substituting into $\varphi(\lambda; q)$ and taking $\inf_\lambda$ enforces $\tilde d - Gm - v = 0$, yielding the **primal feasibility form**:

$$\boxed{h_U(q)\;=\;\sup_{m \in B,\; v \in V}\; \langle T^*q,\, m \rangle
  \quad\text{s.t.}\quad Gm + v = \tilde d.}$$

### 4.2 Why this is computationally attractive

- The feasible set $\{(m,v) : m \in B, v \in V, Gm + v = \tilde d\}$ does **not depend on $q$** — only the linear objective $\langle T^*q, m \rangle$ does. For many directions $q_1, \ldots, q_p$, one can find a feasible $(m^*, v^*)$ once and then evaluate $p$ linear objectives at negligible extra cost.
- Projections onto $B$ and $V$ (balls/ellipsoids) are cheap. The `Subset.project(x)` method in `pygeoinf.subsets` already provides this.

### 4.3 Chambolle–Pock / PDHG

The **Chambolle–Pock algorithm** (Chambolle & Pock, 2011, *J. Math. Imaging Vision* 40:120–145) solves the generic saddle problem

$$\min_x \max_y\; \langle Kx, y\rangle + G(x) - F^*(y)$$

using alternating proximal steps. Applied to our problem (with $K = G$, $G(m,v) = \delta_B(m) + \delta_V(v)$, $F = \delta_{\{\tilde d\}}$ for the equality):

$$y^{n+1} \leftarrow \operatorname{prox}_{\sigma F^*}(y^n + \sigma K \bar x^n),$$
$$x^{n+1} \leftarrow \operatorname{prox}_{\tau G}(x^n - \tau K^* y^{n+1}),$$
$$\bar x^{n+1} \leftarrow x^{n+1} + \theta(x^{n+1} - x^n).$$

**Convergence guarantees (Chambolle & Pock 2011):**
- $O(1/N)$ ergodic rate for the primal-dual gap when $\theta = 1$ and step sizes satisfy $\tau \sigma \|K\|^2 \le 1$.
- $O(1/N^2)$ rate when $G$ is strongly convex with parameter $\gamma > 0$, achieved by updating $\theta_n = 1/\sqrt{1 + 2\gamma\tau_n}$ and shrinking steps accordingly.
- Banert et al. (2023, arXiv:2309.03998) extended the convergence proof to $\theta > 1/2$ with relaxed step condition $\tau\sigma\|K\|^2 < 4/(1+2\theta)$.

**Fit to existing code:**
- $\operatorname{prox}_{\tau G}$ on $B \times V$ = projections via `Subset.project`.
- $K = G$ and $K^* = G^*$ are already `LinearOperator` in `pygeoinf`.
- Equality constraint prox = trivial orthogonal projection onto $\{\tilde d\}$.

### 4.4 ADMM

ADMM on $(m, v)$ with equality constraint $Gm + v = \tilde d$ is structurally equivalent to Douglas–Rachford splitting applied to the dual. Each iteration involves:
- Update $m$: project onto $B$.
- Update $v$: project onto $V$.
- Update dual multiplier $\mu$: gradient ascent on the augmented Lagrangian.

ADMM is often faster in practice than Chambolle–Pock because the augmented Lagrangian provides stronger curvature information, but requires tuning the augmented Lagrangian penalty parameter.

**Proposal:** add a `class PrimalFeasibilitySolver` that accepts `Subset` constraints and a `LinearOperator` equality side, producing feasible $(m, v)$ and, optionally, the dual multiplier $\mu$ (= optimal $\lambda$ for the dual master).

---

## 5. Smoothing + Accelerated Gradient (Fast Path for Ball/Ellipsoid Sets)

When both sets are balls or ellipsoids, the support functions are norms (up to a linear term). This enables **Moreau–Yosida smoothing** of the dual master objective, restoring differentiability and allowing fast gradient methods.

### 5.1 Smoothed ball support

For `BallSupportFunction` with centre $c$ and radius $r$:

$$\sigma_\mathrm{ball}(z) = \langle z, c\rangle + r\|z\|.$$

Smooth the norm with parameter $\varepsilon > 0$:

$$\|z\|_\varepsilon = \sqrt{\|z\|^2 + \varepsilon^2},$$

giving gradient

$$\nabla_z \sigma_{\mathrm{ball},\varepsilon}(z)
  = c + r\,\frac{z}{\sqrt{\|z\|^2 + \varepsilon^2}}.$$

The Lipschitz constant of this gradient (through $z = T^*q - G^*\lambda$) is $L_\varepsilon = r \|G\|^2 / \varepsilon$.

### 5.2 Smoothed ellipsoid support

For ellipsoid support $\sigma(z) = \langle z, c\rangle + r\|A^{-1/2}z\|$:

$$\nabla_z \sigma_\varepsilon(z)
  = c + r\,\frac{A^{-1} z}{\sqrt{\langle z,\, A^{-1} z\rangle + \varepsilon^2}}.$$

### 5.3 Accelerated gradient once smoothed

Once the objective is $L_\varepsilon$-smooth, one can apply:
- **L-BFGS-B** via `scipy.optimize.minimize(method='L-BFGS-B')`: quasi-Newton, often fastest in practice. Zero extra dependencies.
- **Nesterov accelerated gradient / FISTA**: $O(1/k^2)$ convergence, minimal code.

### 5.4 Continuation schedule

The approximation error from smoothing is $O(\varepsilon)$, so final accuracy requires small $\varepsilon$. Use a schedule:

$$\varepsilon_0 \gg \varepsilon_1 \gg \cdots \gg \varepsilon_L \approx \mathrm{tol},$$

solving at each level and warm-starting from the previous minimiser. Typical schedule: $\varepsilon_\ell = \varepsilon_0 \cdot 10^{-\ell}$ for $L = 4{-}6$ levels, starting at $\varepsilon_0 = 10^{-2} \cdot \|\tilde d\|$.

---

## 6. Warm-Starting and the Multi-Direction Case ($P = \mathbb{R}^p$, Tens of Directions)

### 6.1 Why the multi-direction case matters

The dual master is solved once per direction $q \in \mathbb{R}^p$. For $p = 1$ there are two solves. For $p > 1$ one needs $2p$ solves (one per signed unit vector, or more for non-axis-aligned property directions). With $p \sim \mathcal{O}(10)$, warm-starting is important.

### 6.2 Primal warm-start in $\lambda$

`DualMasterCostFunction.set_direction(q)` updates only the cached $T^* q$. For a batch of directions $(q_1, \ldots, q_p)$: use $\lambda^*(q_{i-1})$ as starting point for $q_i$ (sequential sweep). Works well when directions are "nearby". Requires no changes to the solver interface.

### 6.3 Bundle reuse

Cutting-plane cuts are direction-specific: cuts for $q_i$ underestimate $\varphi(\cdot; q_i)$, not $\varphi(\cdot; q_j)$. However:
- The **stability centre** $\hat\lambda$ from the previous direction is a good warm start.
- OSQP's warm-start (`osqp.solve(warm_start=True)`) can reduce the per-iteration QP solve cost at the first bundle iteration.

### 6.4 Parallel evaluation

When directions are independent, solves can be parallelised:

```python
def solve_support_values(
    cost: DualMasterCostFunction,
    qs: list[Vector],
    solver: LevelBundleMethod,
    lambda0: Vector,
    *,
    warm_start: bool = True,
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[Vector], list[BundleResult]]:
    """
    Returns: support_values (p,), optimal_lambdas (p,), diagnostics (p,)
    """
```

When `n_jobs=1`: sequential, with $\lambda$ warm-start across directions.
When `n_jobs > 1`: joblib parallel using `pygeoinf.parallel`; each worker gets a fresh `DualMasterCostFunction` instance.

### 6.5 Multi-direction with primal feasibility (large $p$)

In the primal feasibility form (Section 4), the feasible set is independent of $q$. One primal-dual solve yields $(m^*, v^*)$; all $p$ objectives $\langle T^* q_i, m^* \rangle$ are then free. Optimal for large $p$; requires the primal solver in Section 4 to be implemented first.

---

## 7. API / Class Proposals

### 7.1 Oracle protocol

```python
class HasValueAndSubgradient(Protocol):
    def value_and_subgradient(self, x: Vector) -> tuple[float, Vector]: ...
```

### 7.2 Bundle solvers

| Class | Role |
|---|---|
| `@dataclass Cut` | Stores `(x, f, g, iteration)` |
| `class Bundle` | Manages cuts, computes lower bound, returns constraint matrices |
| `@dataclass BundleResult` | Stores `(x_best, f_best, f_low, gap, converged, n_iters, n_serious)` |
| `class LevelBundleMethod` | Level bundle loop with gap certificate |
| `class ProximalBundleMethod` | Proximal (always-feasible) variant |

Key constructor parameters: `alpha` (level), `tolerance` (gap), `max_iterations`, `bundle_size`, `qp_solver`.

### 7.3 QP solver abstraction

```python
@dataclass
class QPResult:
    x: np.ndarray
    obj: float
    status: str   # 'solved', 'infeasible', ...

class QPSolver(Protocol):
    def solve(self, P, q, A, l, u, x0=None) -> QPResult: ...
```

### 7.4 Direction batch helper

```python
def solve_support_values(
    cost, qs, solver, lambda0,
    *, warm_start=True, n_jobs=1,
) -> tuple[np.ndarray, list[Vector], list[BundleResult]]: ...
```

### 7.5 Primal-dual feasibility solver (future)

```python
class ChambollePockSolver:
    """
    Solves:  max_{m in B, v in V}  <c, m>  s.t.  G m + v = d_tilde
    via Chambolle-Pock (2011).
    Convergence: O(1/N) with tau*sigma*||G||^2 <= 1.
    """
    def solve(self, c: Vector, x0=None) -> tuple[Vector, Vector, float]: ...
```

---

## 8. Dependency Recommendations

| Path | Core deps | Optional deps | When to choose |
|---|---|---|---|
| **In-house SciPy-only** | `numpy`, `scipy` | — | Always available; SLSQP for QP |
| **OSQP** | as above | `osqp` | Best fit for bundle inner loop: ADMM-based, warm-startable |
| **Clarabel** | as above | `clarabel` | Interior-point; native QP handling; no epigraph needed |
| **CVXPY** | as above | `cvxpy` | Prototyping only; dispatches to Clarabel/OSQP/SCS/HiGHS |

**Recommended `pyproject.toml` extras:**

```toml
[project.optional-dependencies]
fast-bundle = ["osqp"]
bundle-alt  = ["clarabel"]
```

---

## 9. Implementation Roadmap

1. `Cut`, `Bundle`, `BundleResult` dataclasses; `QPSolver` protocol; `SciPyQPSolver`.
2. Proximal bundle loop with gap tracking and serious/null steps.
3. Level bundle loop (adds level management on top of proximal).
4. `DualMasterCostFunction.value_and_subgradient`: avoid duplicated oracle work.
5. `OSQPQPSolver`: warm-starting across bundle iterations.
6. `solve_support_values` batch helper: sequential warm-start first, then `n_jobs > 1`.
7. *(Optional)* `ClarabelQPSolver` backend.
8. *(Optional)* Smoothing + L-BFGS-B fast path for ball/ellipsoid sets.
9. *(Optional)* `ChambollePockSolver` for the primal feasibility form.

---

## 10. Practical Parameter Defaults and Robustness Notes

| Setting | Proximal bundle | Level bundle |
|---|---|---|
| $\rho_0$ | `1.0` (×2 null, ÷2 serious) | — |
| $\alpha$ | — | `0.1` or `0.2` |
| `tolerance` (gap $\Delta_k$) | `1e-6` | `1e-6` |
| `bundle_size` | 50–100 | 100–200 |
| `max_iterations` | 500 | 500 |
| Smoothing $\varepsilon_0$ | — | `1e-2 * norm(d_tilde)` |

**Robustness:**
- `SupportFunction.support_point` may return `None`. Proximal bundle methods support $\varepsilon$-subgradients gracefully (Kiwiel 1995); bundle code should handle `None` by falling back to a finite-difference approximation.
- Restrict initial implementations to `EuclideanSpace`; raise `NotImplementedError` for general `HilbertSpace` until `to_components / from_components` round-tripping is fully verified.
- Bundle cuts can become near-parallel (ill-conditioned QP). `Bundle.compress` should remove cuts with small dual multipliers in the master QP solution.

---

## References

- Chambolle, A. & Pock, T. (2011). "A first-order primal-dual algorithm for convex problems with applications to imaging". *J. Math. Imaging Vision* 40:120–145. [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)
- Banert, S. et al. (2023). "Chambolle-Pock revisited: Convergence and refinements." arXiv:2309.03998.
- Kiwiel, K.C. (1990). "Proximity control in bundle methods for convex nondifferentiable minimization". *Math. Program.* 46:105–122.
- Kiwiel, K.C. (1995). "Proximal level bundle methods for convex nondifferentiable optimization, saddle-point problems and variational inequalities". *Math. Program.* 69:89–109.
- Lemaréchal, C., Nemirovskii, A. & Nesterov, Y. (1995). "New variants of bundle methods". *Math. Program.* 69:111–147.
- Nesterov, Y. (1983). "A method of solving a convex programming problem with convergence rate $O(1/k^2)$". *Soviet Math. Doklady* 27:372–376.
- Stellato, B. et al. (2020). "OSQP: an operator splitting solver for quadratic programs". *Math. Program. Comput.* 12:637–672. [osqp.org](https://osqp.org/)
- Goulart, P. & Chen, Y. (2024). Clarabel: An interior-point solver for conic programs. [clarabel.org](https://clarabel.org/)
