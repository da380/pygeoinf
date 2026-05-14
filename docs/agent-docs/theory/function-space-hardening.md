# Function-Space Hardening of Gaussian Priors

> **Math companion** to [`theory/hardening.md`](hardening.md). That document develops the philosophy and a broad menu of hardening choices; this document focuses on the *two probabilistically valid replacements* for the finite-dimensional Mahalanobis ellipsoid in genuinely infinite-dimensional model spaces — the **ambient norm ball** and the **weakened covariance ellipsoid** — and gives the full numerical theory behind each. It is the mathematical reference cited by `pygeoinf.gaussian_measure.GaussianMeasure.credible_set` for `geometry ∈ {"ambient_ball", "weakened_ellipsoid"}`.

---

## 1. Setting and notation

Let $(H, \langle\cdot,\cdot\rangle_H)$ be a real separable Hilbert space and $\mu = \mathcal{N}(m_0, C)$ a Gaussian measure on $H$ with mean $m_0 \in H$ and self-adjoint trace-class positive covariance $C : H \to H$. Let $\{(\lambda_j, e_j)\}_{j\ge 1}$ be the eigenpairs of $C$, ordered $\lambda_1 \ge \lambda_2 \ge \dots > 0$, with $\sum_j \lambda_j = \operatorname{tr}(C) < \infty$. The Karhunen–Loève expansion (cf. [`hardening.md`](hardening.md) §5, eq. (eq:kl)) writes a sample as

$$
X = m_0 + \sum_{j=1}^\infty \sqrt{\lambda_j}\, Z_j\, e_j, \qquad Z_j \stackrel{\text{iid}}{\sim} \mathcal{N}(0,1).
$$

For any positive self-adjoint $A : H \to H$ commuting with $C$ — in particular $A = C^{-\theta}$ for $\theta \in \mathbb{R}$ — define the **gauge**

$$
R_A(m)^2 = \langle A(m - m_0),\, m - m_0 \rangle_H.
$$

A *hardening at level $p$* is a measurable set $S_p \subset H$ with $\mu(S_p) = p$. We restrict attention to *gauge-balls*

$$
S_p = \{m \in H : R_A(m) \le r_p\}.
$$

The hardening problem reduces to two coupled questions:
1. **Probability**: what is the law of $R_A(X)^2$ under $X \sim \mu$?
2. **Computation**: how do we compute its $p$-quantile $r_p^2$ in practice?

The finite-dim case (which is what the original `credible_set` implements) has a clean answer: with $A = C^{-1}$, the gauge is the Mahalanobis distance and $R_A(X)^2 \sim \chi^2_n$. Infinite-dim breaks this in exactly one place — $\sum_j Z_j^2$ diverges — and we must replace $A$ accordingly. The next two sections handle the two replacements.

---

## 2. Ambient norm ball: $A = I$

Choose the simplest gauge $R_I(m) = \|m - m_0\|_H$. By KL,

$$
R_I(X)^2 = \sum_{j=1}^\infty x_j^2 = \sum_{j=1}^\infty \lambda_j Z_j^2, \tag{2.1}
$$

with $x_j = \langle X - m_0, e_j\rangle_H = \sqrt{\lambda_j} Z_j$.

**Trace condition.** $R_I(X)^2 < \infty$ a.s. iff $\sum_j \lambda_j < \infty$. This is *already guaranteed* by $C$ being trace-class. So the ambient ball *always* admits a finite calibrated radius.

Define

$$
F_{\text{amb}}(t) = \mathbb{P}\!\left(\sum_{j=1}^\infty \lambda_j Z_j^2 \le t\right).
$$

$F_{\text{amb}}$ is a generalized chi-square CDF with weights $w_j = \lambda_j > 0$. The probability-calibrated radius is

$$
r_p^2 = F_{\text{amb}}^{-1}(p). \tag{2.2}
$$

**Degenerate isotropic case.** If $\lambda_1 = \cdots = \lambda_n = \sigma^2$ and $\lambda_j = 0$ for $j > n$, then $F_{\text{amb}}(t) = \mathbb{P}(\chi^2_n \le t/\sigma^2)$ and $r_p^2 = \sigma^2 \chi^2_n(p)$, recovering the finite-dim chi-square formula. In general $F_{\text{amb}}$ does *not* reduce to a single chi-square, which is precisely why the existing `chi2.ppf(p, k)` calibration in `credible_set` is the wrong default for genuine function spaces.

---

## 3. Weakened covariance ellipsoid: $A = C^{-\theta}$, $\theta \in [0,1]$

Choose $A = C^{-\theta}$ for some $\theta \in [0, 1]$. The gauge becomes

$$
R_\theta(m)^2 = \|C^{-\theta/2}(m - m_0)\|_H^2 = \sum_{j=1}^\infty \lambda_j^{-\theta} x_j^2.
$$

Under $X \sim \mu$,

$$
R_\theta(X)^2 = \sum_{j=1}^\infty \lambda_j^{-\theta} \cdot \lambda_j Z_j^2 = \sum_{j=1}^\infty \lambda_j^{1-\theta} Z_j^2. \tag{3.1}
$$

**(1−θ)-trace condition.** $R_\theta(X)^2 < \infty$ a.s. iff

$$
\sum_{j=1}^\infty \lambda_j^{1-\theta} < \infty. \tag{3.2}
$$

This is *strictly stronger* than the trace condition $\sum \lambda_j < \infty$ when $\theta \in (0, 1)$, because $\lambda_j^{1-\theta} \ge \lambda_j$ once $\lambda_j \le 1$ (which is the regime $j$ large). For analytic spectra such as $\lambda_j \propto j^{-\alpha}$ for some $\alpha > 1$, condition (3.2) holds iff $\alpha (1-\theta) > 1$, i.e.,

$$
\theta < 1 - 1/\alpha.
$$

For example, the inverse Laplacian on $[0,1]$ has $\lambda_j \propto j^{-2}$ ($\alpha = 2$), giving the upper bound $\theta < 1/2$. For $\theta \in [1/2, 1)$ on this spectrum, $\sum \lambda_j^{1-\theta}$ diverges and a calibrated radius does *not* exist in the genuine infinite-dim limit. The implementation emits a `UserWarning` when the tail of $\lambda_j^{1-\theta}$ is detected to be slowly decaying after truncation (see §5).

**Endpoints.**
- $\theta = 0$ recovers the ambient ball: $R_0 = R_I$, equation (2.1) = (3.1).
- $\theta = 1$ gives the **Cameron–Martin** norm:

  $$
  R_1(X)^2 = \sum_{j=1}^\infty Z_j^2 = \infty \quad \text{a.s.}
  $$

  by the strong law of large numbers ([`hardening.md`](hardening.md) §5.2, eq. (eq:CM-zero)). The Cameron–Martin ball has Gaussian measure zero in genuine infinite-dim. The existing `credible_set(..., geometry="cameron_martin")` is consistent only because it implicitly truncates the spectrum to finite `rank`. The new `geometry="weakened_ellipsoid"` with $\theta < 1$ is the right replacement that retains covariance-adapted geometry *and* probability mass.

Define

$$
F_\theta(t) = \mathbb{P}\!\left(\sum_{j=1}^\infty \lambda_j^{1-\theta} Z_j^2 \le t\right).
$$

The probability-calibrated radius is

$$
r_p^2 = F_\theta^{-1}(p). \tag{3.3}
$$

**Geometric interpretation.** $R_\theta$ interpolates between the *isotropic* ambient norm ($\theta=0$, weights $\lambda_j$ — high-eigenvalue directions dominate the radius) and the *whitened* Cameron–Martin norm ($\theta=1$, weights $1$ — all directions contribute equally, divergent in infinite-dim). Intermediate $\theta$ retains *part* of the covariance anisotropy while damping the high-frequency tail just enough to keep the gauge integrable.

---

## 4. Weighted chi-square: distribution theory

Both calibration problems reduce to: given weights $w = (w_j)_{j=1}^N$ with $w_j > 0$ and $\sum_j w_j < \infty$, compute either

$$
F_Q(t) = \mathbb{P}(Q \le t), \quad Q = \sum_{j=1}^N w_j Z_j^2 \quad \text{with } Z_j \stackrel{\text{iid}}{\sim} \mathcal{N}(0,1),
$$

or its inverse $F_Q^{-1}(p)$. Throughout this section the truncation $N$ is treated as a finite — but possibly large — number. Section 5 discusses truncation bias.

### 4.1 Characteristic function

$Q$ is a finite sum of independent rescaled chi-square-1 random variables, so

$$
\varphi_Q(s) = \mathbb{E}[e^{isQ}] = \prod_{j=1}^N (1 - 2is w_j)^{-1/2}, \quad s \in \mathbb{R}. \tag{4.1}
$$

### 4.2 Imhof's method (default, exact)

The Gil-Pelaez inversion formula combined with the half-line decomposition gives, for $t > 0$,

$$
\mathbb{P}(Q \le t) = \tfrac12 - \frac{1}{\pi} \int_0^\infty \frac{\sin\theta(u)}{u\,\rho(u)}\, du, \tag{4.2}
$$

where

$$
\theta(u) = \tfrac12 \sum_{j=1}^N \arctan(w_j u) - \tfrac12 t u, \qquad
\rho(u) = \prod_{j=1}^N (1 + w_j^2 u^2)^{1/4}.
$$

The integrand is well-behaved: it oscillates rapidly for large $u$ but the denominator $u \rho(u)$ damps it as $u \to \infty$ since $\rho(u) \ge \prod_j (w_j u)^{1/2}$. For $u \to 0^+$ the integrand is bounded since $\sin\theta(u) \sim \theta(u) = O(u)$ near zero. Implementation: a single `scipy.integrate.quad` call delivers $F_Q(t)$ to machine precision; root-finding via `scipy.optimize.brentq` recovers $F_Q^{-1}(p)$.

**Cost.** $O(N)$ per integrand evaluation. Typical quadrature uses $O(50)$ nodes, so each CDF call is $O(50 N)$ flops. For $N$ up to $\sim 10^4$ this is far cheaper than alternatives.

### 4.3 Welch–Satterthwaite (moment matching)

Match the first two moments of $Q$ to a scaled chi-square $a \chi^2_\nu$:

$$
\mathbb{E}[Q] = \sum_j w_j, \qquad \operatorname{Var}(Q) = 2 \sum_j w_j^2.
$$

Setting $\mathbb{E}[a\chi^2_\nu] = a\nu$ and $\operatorname{Var}(a\chi^2_\nu) = 2 a^2 \nu$,

$$
a = \frac{\sum_j w_j^2}{\sum_j w_j}, \qquad \nu = \frac{(\sum_j w_j)^2}{\sum_j w_j^2}. \tag{4.3}
$$

Then $r_p^2 \approx a\,\chi^2_\nu(p)$.

**When it's accurate.** Best when the $w_j$ are roughly equal: $\nu \to N$ and the approximation becomes exact for equal weights (since $Q$ literally is $a\chi^2_N$). Degrades when the weights are highly anisotropic, e.g., one $w_1 \gg w_j$ for $j > 1$: then $\nu \to 1$ and the approximation underestimates tail probabilities.

**Use cases.** Free initial bracket for Imhof's bisection; fast preliminary radius for benchmarking; default when $N$ is so large that Imhof's $O(N)$ cost dominates.

### 4.4 Lugannani–Rice saddlepoint

The cumulant generating function of $Q$ is

$$
K_Q(s) = \log \mathbb{E}[e^{sQ}] = -\tfrac12 \sum_{j=1}^N \log(1 - 2sw_j), \quad s < \frac{1}{2 \max_j w_j}.
$$

The saddlepoint $\hat s(t)$ solves $K_Q'(\hat s) = t$, i.e., $\sum_j \frac{w_j}{1 - 2\hat s w_j} = t$. Define

$$
\hat w = \operatorname{sgn}(\hat s)\sqrt{2(\hat s\, t - K_Q(\hat s))}, \qquad \hat u = \hat s \sqrt{K_Q''(\hat s)},
$$

where $K_Q''(s) = 2 \sum_j \frac{w_j^2}{(1 - 2s w_j)^2}$. The Lugannani–Rice approximation is

$$
\mathbb{P}(Q \le t) \approx \Phi(\hat w) + \phi(\hat w)\left(\frac{1}{\hat w} - \frac{1}{\hat u}\right), \tag{4.4}
$$

with $\Phi, \phi$ the standard normal CDF/PDF.

**Strength.** Excellent in the deep tails ($p \in \{0.99, 0.999\}$): relative error often $O(10^{-4})$ where WS already loses 1–2 digits. Cost: solving the saddlepoint equation is one root-find per evaluation.

### 4.5 Monte Carlo (reference)

Draw $M$ samples $q^{(i)} = \sum_j w_j (Z_j^{(i)})^2$ with independent standard normals, return $\text{quantile}_p(\{q^{(i)}\})$. Sampling error $O(1/\sqrt{M})$ per quantile; bias zero. Use $M = 10^5$ for unit tests and as a fallback when Imhof's quadrature fails (e.g., extreme anisotropy with $w_1 / w_N > 10^{12}$).

### 4.6 Recommended defaults

| Regime | Default method | Reason |
|---|---|---|
| $N \le 10^3$, modest anisotropy | Imhof | Machine-precision quantiles at small cost |
| $N > 10^4$, modest anisotropy | WS, refine with Imhof | Closed-form is cheaper, refine only if needed |
| $p > 0.99$ | Saddlepoint or Imhof | WS loses tail accuracy |
| Cross-validation | MC ($M = 10^5$) | Unbiased reference |

The implementation supports all four via the `quantile_method` argument; Imhof is the public default.

---

## 5. Truncation bias

In practice we work with a finite spectrum $\{\lambda_j\}_{j=1}^N$ (analytic truncation, or output of `LowRankEig.from_randomized`). Define the truncated gauge

$$
R_\theta^{(N)}(X)^2 = \sum_{j=1}^N \lambda_j^{1-\theta} Z_j^2.
$$

The truncation bias is

$$
0 \le R_\theta(X)^2 - R_\theta^{(N)}(X)^2 = \sum_{j>N} \lambda_j^{1-\theta} Z_j^2,
$$

with expected value $\mathbb{E}[\text{bias}] = \sum_{j > N} \lambda_j^{1-\theta}$. For analytic spectra $\lambda_j \asymp j^{-\alpha}$,

$$
\sum_{j > N} j^{-\alpha(1-\theta)} = O(N^{-(\alpha(1-\theta) - 1)}),
$$

valid whenever $\alpha(1-\theta) > 1$. For example on the inverse Laplacian ($\alpha = 2$) and $\theta = 0.4$, the tail mean decays as $O(N^{-0.2})$ — slow but convergent. For $\theta = 0.5$ the series is borderline; for $\theta > 0.5$ it diverges and no finite radius exists in the limit.

**Practical diagnostic.** Compute radii $r_p^{(N)}$ and $r_p^{(2N)}$. If $|r_p^{(2N)} - r_p^{(N)}| / r_p^{(N)}$ is small (e.g., $< 10^{-3}$), accept $r_p^{(2N)}$. Else increase $N$ or warn the user that the (1−θ)-trace condition is borderline.

**Monotonicity.** Adding eigenvalues to the sum can only increase $R_\theta^{(N)}(X)^2$ pointwise, hence $F_\theta^{(N)}$ stochastically dominates $F_\theta^{(N+1)}$. Equivalently $r_p^{(N)}$ is monotonically non-decreasing in $N$. This gives a one-sided convergence guarantee.

---

## 6. Lanczos matrix functions: applying $C^{-\theta/2}$ without a spectrum

The weakened gauge $R_\theta(m) = \|C^{-\theta/2}(m - m_0)\|_H$ requires the operator $C^{-\theta/2}$ applied to a vector. When the spectrum is known, the spectral path of §3 evaluates this via $\sum_j \lambda_j^{-\theta/2} \langle e_j, v\rangle e_j$. When the spectrum is *not* known, we must compute $C^{-\theta/2} v$ matrix-free. This section develops the **Lanczos matrix-function** method.

### 6.1 Krylov subspaces and the Lanczos process

For self-adjoint positive $C$ and starting vector $v \in H \setminus \{0\}$, the order-$k$ Krylov subspace is

$$
\mathcal{K}_k(C, v) = \operatorname{span}\{v,\, Cv,\, C^2v, \ldots,\, C^{k-1}v\}.
$$

Apply Gram–Schmidt to the Krylov basis: set $q_1 = v / \|v\|_H$, and for $j = 1, 2, \ldots, k-1$:

$$
\begin{aligned}
\alpha_j &= \langle C q_j,\, q_j \rangle_H, \\
r_j &= C q_j - \alpha_j q_j - \beta_{j-1} q_{j-1}, \\
\beta_j &= \|r_j\|_H, \\
q_{j+1} &= r_j / \beta_j,
\end{aligned}
\tag{6.1}
$$

with $\beta_0 = 0$, $q_0 = 0$. The vectors $\{q_j\}$ are orthonormal in $H$ and the matrix of $C$ restricted to $\mathcal{K}_k(C,v)$ in this basis is the symmetric tridiagonal

$$
T_k = \begin{pmatrix} \alpha_1 & \beta_1 & & \\ \beta_1 & \alpha_2 & \beta_2 & \\ & \beta_2 & \ddots & \ddots \\ & & \ddots & \alpha_k \end{pmatrix} \in \mathbb{R}^{k\times k}.
$$

Let $V_k : \mathbb{R}^k \to H$ be the operator with columns $q_1, \ldots, q_k$. The **Lanczos identity** is

$$
C V_k = V_k T_k + \beta_k\, q_{k+1}\, e_k^\top. \tag{6.2}
$$

### 6.2 Matrix functions on Krylov subspaces

For any $f$ analytic on the spectrum of $C$,

$$
f(C) v \approx \|v\|_H \cdot V_k\, f(T_k)\, e_1, \tag{6.3}
$$

where $f(T_k)$ is computed by diagonalizing $T_k = S \Theta S^\top$ (cost $O(k^3)$ via `np.linalg.eigh`) and forming

$$
f(T_k) = S\,\operatorname{diag}(f(\theta_1), \ldots, f(\theta_k))\,S^\top.
$$

For our application $f(\lambda) = \lambda^{-\theta/2}$ with $\theta \in (0, 1)$, $f$ is analytic on $(0, \infty)$. The Ritz values $\theta_i$ approximate the eigenvalues of $C$ (the extremes first); since $C \succ 0$ in exact arithmetic and the Krylov subspace is invariant under $C$, the $\theta_i$ are all positive and $f(\theta_i)$ is well-defined.

### 6.3 Convergence and error bounds

For analytic $f$ on a domain containing the spectrum $[\lambda_{\min}, \lambda_{\max}] \subset (0, \infty)$,

$$
\|f(C) v - \|v\|_H V_k f(T_k) e_1\|_H \le C(f) \rho^k, \tag{6.4}
$$

with $\rho = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$ and $\kappa = \lambda_{\max}/\lambda_{\min}$ (Saad-style estimate; see e.g. Higham *Functions of Matrices*, Thm. 13.5). Practically: convergence is fast on well-conditioned $C$, and the dominant high-eigenvalue directions are captured first.

**Caveat: trace-class $C$.** For genuine trace-class covariances, $\lambda_{\min}$ does not exist (the spectrum accumulates at 0), so $\kappa = \infty$ and the bound (6.4) degenerates. In practice this is harmless because the Krylov process captures the large-eigenvalue directions first; the tail (small $\lambda_j$) contributes very little to $f(C) v$ for the negative-power $f$ we care about (since $\lambda^{-\theta/2}$ is largest where $\lambda$ is smallest, but it multiplies the small projection $\langle e_j, v\rangle$ which is itself $O(\sqrt{\lambda_j})$ for random $v$). A rigorous error analysis is beyond this companion; we rely on numerical convergence diagnostics in the implementation.

### 6.4 Floating-point reality: loss of orthogonality

In floating-point arithmetic, the recurrence (6.1) gradually loses orthogonality of the $\{q_j\}$. After roughly $k \sim O(\sqrt{\epsilon^{-1}})$ steps with no reorthogonalisation (here $\epsilon$ the machine eps), $T_k$ no longer represents the restriction of $C$ to the (true) Krylov subspace. Three remedies:

- **Full reorthogonalisation**: after computing $r_j$, subtract $\sum_{i \le j} \langle r_j, q_i\rangle q_i$ explicitly. Cost: $O(k)$ extra inner products per step, $O(k^2)$ total. *Default for the pygeoinf implementation.*
- **Selective / partial reorthogonalisation**: monitor a Paige-style orthogonality bound and only reorthogonalise when it drifts. Saves work for large $k$ but adds complexity. Not implemented in v1.
- **No reorthogonalisation**: cheap but unreliable for $k > 30$ or so. Provided for benchmarking only.

### 6.5 Breakdown and fallback

$\beta_j = 0$ for some $j \le k$ indicates that the Krylov subspace has stabilised (an invariant subspace has been found). This is *success* — truncate and return the exact answer from $T_j$.

Practical failure modes for $f(\lambda) = \lambda^{-\theta/2}$:
1. Floating-point indefiniteness: a Ritz value $\theta_i$ becomes numerically zero or negative due to rounding. Detect by checking the smallest eigenvalue of $T_k$ against a tolerance.
2. Inadequate Krylov dimension: the starting vector $v$ is poorly aligned with the eigenvectors carrying significant $\lambda^{-\theta/2}$ weight. Symptoms: $\|f(C) v\|$ much smaller than expected.

In either case the implementation falls back to **randomized low-rank**: call `LowRankEig.from_randomized(self.covariance, k, measure=self)` to obtain an approximate eigendecomposition $(U, \Lambda)$, then evaluate $f(C) v \approx U f(\Lambda) U^\top v$. This is robust because randomized range-finding with power iteration aggregates information from multiple test vectors and is less sensitive to the alignment of any single $v$.

### 6.6 Spectral vs. Lanczos: a comparison

| Aspect | Spectral / `LowRankEig` | Lanczos per-sample |
|---|---|---|
| Requires user-supplied spectrum | Optional (auto-computable) | No |
| Setup cost | $O(N \cdot \operatorname{cost}(C\cdot v))$ once | None |
| Per-gauge-evaluation cost | $O(N)$ | $O(k \cdot \operatorname{cost}(C\cdot v))$ |
| Reuses across samples | Yes | No (one Lanczos per sample) |
| Accuracy control | Truncation rank $N$ | Krylov dim $k$, reorth strategy |
| Best when | Many samples needed | One-off evaluations, very large $C$ |
| Failure mode | Truncation bias for slow $\lambda_j^{1-\theta}$ decay | Loss of orthogonality, FP-indefiniteness |

For the `credible_set` API both paths are exposed via `fractional_apply ∈ {"auto", "lanczos", "low_rank_eig"}`. The default `"auto"` resolves: if a spectrum is available, use it (cheapest); otherwise Lanczos for single-shot, randomized eig for batches.

---

## 7. End-to-end calibration pipeline

For a Gaussian measure $\mu$ on $H$, requested probability $p$, and either $\text{geometry} = \text{ambient\_ball}$ or $\text{weakened\_ellipsoid}(\theta)$:

1. **Resolve a spectrum**: $\{\lambda_j\}_{j=1}^N$ from one of {user-supplied array, `LowRankEig` instance, callable, randomized fallback}. (Skippable if the radius-method is purely sampling-based and the gauge is the ambient norm.)
2. **Form weights**: $w_j = \lambda_j$ (ambient) or $w_j = \lambda_j^{1-\theta}$ (weakened).
3. **Compute $r_p^2 = F_w^{-1}(p)$** via the chosen `quantile_method` (Imhof / WS / saddlepoint / MC).
4. **Construct the set**:
   - Ambient ball: `Ball(domain=H, center=m_0, radius=r_p)`.
   - Weakened ellipsoid: `Ellipsoid(domain=H, center=m_0, radius=r_p, operator=C^{-θ}, inverse_operator=C^θ, inverse_sqrt_operator=C^{θ/2})`, with the fractional operators implemented matrix-free via §6.

**Sampling-based variant of step 3.** Bypass §4 entirely: draw $M$ samples $X^{(i)} \sim \mu$, compute $R(X^{(i)})^2$ for each (using §6 to apply $C^{-\theta/2}$ if needed), return $\text{quantile}_p(\{R^2\})$. Bias: $O(1/\sqrt{M})$ statistical, zero systematic.

**Self-consistency check.** When both paths are feasible, compute both radii and require them to agree within a documented tolerance (used in tests).

---

## 8. Connection to existing pygeoinf semantics

The current finite-rank implementation of `credible_set` ([`pygeoinf/gaussian_measure.py`](../../../pygeoinf/gaussian_measure.py) lines 428–522) is the **special case** of this framework where:

- All eigenvalues are treated as equal: weights $w_j \equiv 1$.
- $r_p^2 = \chi^2_k(p)$ from the standard chi-square table.
- $A = C^{-1}$ exactly (i.e., $\theta = 1$ in our notation).

This is bit-for-bit consistent with the new spectrum-aware path when:
- `spectrum = np.ones(k)`, `theta = 1`, `geometry = "weakened_ellipsoid"`: weighted-chi-square with weights $\lambda_j^{1-1} = 1$ for all $j$, i.e., $\chi^2_k$.

The two API paths are therefore mathematically *aligned*: the new modes do not replace the chi-square Mahalanobis calibration, they *extend* it to spectrum-aware regimes where the equal-weights assumption is wrong.

---

## 9. References

Internal:
- [`hardening.md`](hardening.md) — the parent theoretical document.

External:
- Imhof, J. P. (1961). *Computing the distribution of quadratic forms in normal variables.* Biometrika 48, 419–426.
- Welch, B. L. (1938). *The significance of the difference between two means when the population variances are unequal.* Biometrika 29, 350–362.
- Lugannani, R., Rice, S. O. (1980). *Saddlepoint approximation for the distribution of the sum of independent random variables.* Adv. Appl. Probab. 12, 475–490.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM. (Chapter 6 on Lanczos.)
- Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*, SIAM.
- Bogachev, V. I. (1998). *Gaussian Measures*, AMS Math. Surveys & Monographs 62.
- Stuart, A. M. (2010). *Inverse problems: a Bayesian perspective.* Acta Numerica 19, 451–559.
