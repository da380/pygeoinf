\documentclass[11pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}

% -----------------------------------------------------------------------------
% Notation.  These macros are chosen to be compatible with the notation used in
% the surrounding thesis.  Adjust names locally if your master file already
% defines them.
% -----------------------------------------------------------------------------
\newcommand{\modelspace}{\mathcal{M}}
\newcommand{\dataspace}{\mathcal{D}}
\newcommand{\propertyspace}{\mathcal{Z}}
\newcommand{\SigmaM}{\Sigma_{\modelspace}}
\newcommand{\SigmaD}{\Sigma_{\dataspace}}
\newcommand{\mprior}{\mu_0}
\newcommand{\mpost}{\mu^d}
\newcommand{\mdata}{\mu_{\dataspace}}
\newcommand{\Smodel}{\mathsf{M}}
\newcommand{\Smodelzero}{\mathsf{M}_0}
\newcommand{\Snoise}{\mathsf{E}}
\newcommand{\Sdata}{\mathsf{D}}
\newcommand{\dobs}{d^{\mathrm{obs}}}
\newcommand{\dd}{\,\mathrm{d}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Prob}{\mathbb{P}}
\newcommand{\ip}[2]{\left\langle #1,#2\right\rangle}
\newcommand{\norm}[1]{\left\lVert #1\right\rVert}
\newcommand{\esssupp}{\operatorname{ess\,supp}}
\newcommand{\supp}{\operatorname{supp}}
\newcommand{\trace}{\operatorname{tr}}
\newcommand{\rank}{\operatorname{rank}}
\newcommand{\Range}{\operatorname{Ran}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\Law}{\operatorname{Law}}
\newcommand{\KL}{\mathrm{KL}}
\newcommand{\CM}{\mathrm{CM}}
\newcommand{\HPD}{\mathrm{HPD}}
\newcommand{\id}{\mathrm{Id}}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\newtheorem{remark}{Remark}
\newtheorem{principle}{Principle}
\newtheorem{recipe}{Recipe}

\theoremstyle{plain}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}

\title{Hardening Soft Priors: Gaussian Measures, Quadratic Geometry, and Deterministic Admissible Sets}
\author{Draft note}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This note explains how a probabilistic prior measure can be converted, or
``hardened,'' into deterministic admissible sets.  The emphasis is on Gaussian
measures.  In finite dimensions the construction appears almost canonical: a
Gaussian density with respect to Lebesgue measure has ellipsoidal highest-density
regions, and these are also minimum-Lebesgue-volume sets among sets of the same
probability content.  In infinite-dimensional function spaces this equivalence
breaks.  There is no canonical Lebesgue measure, Gaussian measures do not have
Lebesgue densities, and the Cameron--Martin ellipsoid that most closely resembles
the finite-dimensional Mahalanobis ellipsoid typically has Gaussian measure zero.
Nevertheless, Gaussian measures still provide useful deterministic sets through
ambient norm balls, finite-dimensional truncations, weakened covariance-weighted
ellipsoids, or posterior/property-space credible sets.  The key point is that
hardening is not canonical: it requires a modelling choice of gauge, norm, loss,
or admissible shape class.  This is the reverse of the hard-to-soft conversion
criticised by Backus, and the same lesson appears in both directions: converting
between hard bounds and soft probability distributions necessarily adds or
removes information.
\end{abstract}

\tableofcontents

\section{Motivation}

A Bayesian formulation starts with a soft prior: a probability measure on a model
space.  A deterministic linear inference (DLI) or set-theoretic formulation starts
instead with a hard admissible set.  The problem considered here is the reverse of
Backus' well-known problem of softening hard prior bounds.  We ask:
\begin{quote}
Given a prior probability measure \(\mprior\) on a model space \(\modelspace\),
can one choose a deterministic prior set \(\Smodelzero\subset\modelspace\) in a
principled way?
\end{quote}
For a chosen probability content \(p\in(0,1)\), one might want
\begin{equation}
        \mprior(\Smodelzero)=p.
\end{equation}
But this condition alone is far too weak.  There are infinitely many sets with the
same probability mass.  Even requiring the set to be centred at the prior mean and
convex does not, in general, select a unique set.

In finite-dimensional Gaussian problems, one often resolves this ambiguity by
choosing a highest posterior density or highest prior density region.  With
Lebesgue measure as the reference measure, Gaussian density superlevel sets are
ellipsoids.  They are also minimum-volume sets among sets of fixed Gaussian mass.
This makes the hardening look canonical.

In function spaces, this finite-dimensional story breaks down.  The model space
may be a Banach or Hilbert space of functions, but there is no translation-invariant
Lebesgue measure on it.  Hence there is no canonical notion of ``minimum volume,''
and density level sets depend on the chosen reference measure.  Gaussian measures
still have covariance operators and Cameron--Martin geometry, but their typical
samples do not generally lie in the Cameron--Martin space.  Thus the most obvious
infinite-dimensional Mahalanobis ball usually has probability zero.

This note develops a careful replacement for the finite-dimensional story.  The
main conclusion is:
\begin{quote}
A Gaussian prior does not determine a unique deterministic admissible set in an
infinite-dimensional model space.  It determines several natural structures: a
probability measure on an ambient space, a covariance operator, a
Cameron--Martin geometry, and finite-dimensional projections.  A hardening
operation must choose which of these structures is to be retained.
\end{quote}

\section{Soft and hard prior information}

Let \(\modelspace\) be a model space.  A \emph{soft prior bound} is a probability
measure
\begin{equation}
        \mprior \in \mathcal{P}(\modelspace),
\end{equation}
which expresses relative plausibility over possible models.  A \emph{hard prior
bound} is a deterministic statement such as
\begin{equation}
        m \in \Smodelzero,
        \qquad \Smodelzero\subset\modelspace,
\end{equation}
or, more specifically, a quadratic inequality
\begin{equation}
        Q(m-m_0,m-m_0) \le r^2.
\end{equation}
Backus distinguishes these two types of prior information: a soft bound is a
probability distribution on the model space, while a hard bound is an inequality,
often linear or quadratic \citep{backus1988hardsoft,backus1989csi}.  Backus'
central warning is that replacing a hard quadratic bound by a probability
measure can introduce additional information not contained in the original
inequality.  The present note examines the reverse operation: replacing a
probability measure by a hard set.

\begin{definition}[Hardening]
Let \((\modelspace,\SigmaM)\) be a measurable model space and let
\(\mprior\) be a prior probability measure.  A \emph{hardening} of \(\mprior\) at
probability level \(p\in(0,1)\) is a rule selecting a measurable set
\begin{equation}
        \Smodelzero(p)\in\SigmaM,
        \qquad
        \mprior(\Smodelzero(p))\ge p.
\end{equation}
If equality holds, the set is said to be \emph{probability calibrated}.
\end{definition}

This definition is deliberately broad.  It does not specify which set to choose.
That missing choice is the essential modelling issue.

\section{Finite-dimensional Gaussian hardening}

\subsection{Gaussian density and HPD regions}

Let
\begin{equation}
        X\sim \mathcal{N}(m_0,C)
        \qquad \text{on } \R^n,
\end{equation}
where \(C\) is symmetric positive definite.  With respect to Lebesgue measure,
the density is
\begin{equation}
        \rho(x)
        =
        (2\pi)^{-n/2}\det(C)^{-1/2}
        \exp\left[-\frac12 (x-m_0)^T C^{-1}(x-m_0)\right].
\end{equation}
The density superlevel sets are the Mahalanobis ellipsoids
\begin{equation}
        E_r
        =
        \left\{x\in\R^n:
        (x-m_0)^T C^{-1}(x-m_0)\le r^2
        \right\}.
\end{equation}
If \(Y=C^{-1/2}(X-m_0)\), then \(Y\sim\mathcal{N}(0,I_n)\), and therefore
\begin{equation}
        (X-m_0)^T C^{-1}(X-m_0)
        =
        \norm{Y}^2
        \sim \chi^2_n.
\end{equation}
Thus the probability-calibrated ellipsoid of mass \(p\) is
\begin{equation}
        E_p
        =
        \left\{x:
        (x-m_0)^T C^{-1}(x-m_0)\le \chi^2_n(p)
        \right\},
\end{equation}
where \(\chi^2_n(p)\) denotes the \(p\)-quantile of the chi-square distribution
with \(n\) degrees of freedom.

\subsection{Why the finite-dimensional construction feels canonical}

The finite-dimensional construction is unusually satisfying because several
independent ideas coincide:
\begin{enumerate}[label=(\roman*)]
\item \(E_p\) is a density superlevel set.
\item \(E_p\) is centred at the Gaussian mean.
\item \(E_p\) is an ellipsoid whose axes are determined by the covariance.
\item \(E_p\) has probability \(p\).
\item Among measurable sets of probability \(p\), \(E_p\) has minimal Lebesgue
volume, up to null sets.
\end{enumerate}
The last statement is the rearrangement principle behind highest-density regions:
if a density is largest near the mean and decreases along Mahalanobis rays, then
to capture probability mass with least reference volume one should include the
highest-density points first.

This is the finite-dimensional ideal of hardening:
\begin{equation}
        \text{soft Gaussian prior}
        \quad \leadsto \quad
        \text{minimum-volume HPD ellipsoid of mass } p.
\end{equation}

\subsection{Ambient balls versus covariance ellipsoids}

Even in finite dimensions one may distinguish two different balls:
\begin{align}
        B_r^{\mathrm{amb} }
        &:= \{x:\norm{x-m_0}_2\le r\}, \\
        B_r^{\mathrm{Mah} }
        &:= \{x:\norm{C^{-1/2}(x-m_0)}_2\le r\}.
\end{align}
Both have positive Gaussian probability, and one can choose a radius for either.
The second follows the Gaussian density contours; the first only uses the ambient
Euclidean geometry.  In finite dimensions this distinction is mild because all norms
are equivalent and both sets are full-dimensional.  In infinite dimensions the same
distinction becomes decisive.

\section{Gaussian measures in Hilbert spaces}

Let \(H\) be a separable Hilbert space, for example \(L^2(\Omega)\).  A centred
Gaussian measure \(\mu=\mathcal{N}(0,C)\) on \(H\) is determined by a
self-adjoint non-negative trace-class covariance operator \(C:H\to H\).  If
\begin{equation}
        C e_j = \lambda_j e_j,
        \qquad
        \lambda_j>0,
        \qquad
        \sum_{j=1}^\infty \lambda_j < \infty,
\end{equation}
then a Gaussian random element has the Karhunen--Loeve expansion
\begin{equation}
        X = \sum_{j=1}^\infty \sqrt{\lambda_j}\, Z_j e_j,
        \qquad
        Z_j\sim\mathcal{N}(0,1)\text{ i.i.d.}
        \label{eq:kl}
\end{equation}
The trace-class condition ensures
\begin{equation}
        \E \norm{X}_H^2
        =
        \sum_{j=1}^\infty \lambda_j
        <\infty,
\end{equation}
and hence \(X\in H\) almost surely.

Bogachev's survey records the standard Hilbert-space formulation: Gaussian
measures on a separable Hilbert space are described by a mean vector and a
symmetric non-negative nuclear covariance operator \citep{bogachev1996gaussian}.
This is the infinite-dimensional replacement for the covariance matrix in
\(\R^n\).

\subsection{Ambient norm balls}

The ambient norm ball
\begin{equation}
        B_r^H = \{m\in H:\norm{m-m_0}_H\le r\}
\end{equation}
has probability
\begin{equation}
        \mu(B_r^H)
        =
        \Prob\left(\sum_{j=1}^\infty \lambda_j Z_j^2\le r^2\right).
        \label{eq:ambient-small-ball}
\end{equation}
This is a genuine probability.  In the language of Gaussian process theory it is a
small-ball probability.  It usually has no elementary closed form, but it can be
computed by spectral truncation, characteristic-function inversion, saddlepoint
methods, moment matching, or Monte Carlo.

Thus ambient balls provide legitimate probability-calibrated hardenings:
\begin{equation}
        \Smodelzero(p)
        =
        \{m:\norm{m-m_0}_H\le r_p\},
        \qquad
        \mu(\Smodelzero(p))=p.
\end{equation}
However, the ambient ball may not express the covariance-induced smoothness
well.  If the covariance suppresses high-frequency modes, an \(L^2\)-ball does not
penalize high-frequency and low-frequency directions according to their prior
variances.

\section{The Cameron--Martin space}

\subsection{Definition and geometry}

The Cameron--Martin space associated with \(\mu=\mathcal{N}(0,C)\) is
\begin{equation}
        \mathcal{H}_C = \Range(C^{1/2}),
\end{equation}
with norm
\begin{equation}
        \norm{h}_{\CM}^2
        =
        \norm{C^{-1/2}h}_H^2.
\end{equation}
In the eigenbasis of \(C\), if \(h=\sum_j h_j e_j\), then
\begin{equation}
        \norm{h}_{\CM}^2
        =
        \sum_{j=1}^\infty \frac{h_j^2}{\lambda_j}.
\end{equation}
The corresponding Cameron--Martin ball is
\begin{equation}
        B_r^{\CM}
        =
        \left\{h\in\mathcal{H}_C:
        \sum_{j=1}^\infty \frac{h_j^2}{\lambda_j}\le r^2
        \right\}.
\end{equation}
This is precisely the infinite-dimensional analogue of the finite-dimensional
Mahalanobis ellipsoid.  Formally,
\begin{equation}
        B_r^{\CM}=C^{1/2}\{u\in H:\norm{u}_H\le r\}.
\end{equation}
It is the image of an ambient Hilbert ball under \(C^{1/2}\).  Thus it really does
``follow the contours'' of the covariance geometry.

\subsection{Why typical Gaussian samples are not Cameron--Martin elements}

Substituting the Karhunen--Loeve expansion \eqref{eq:kl} into the
Cameron--Martin norm gives
\begin{equation}
        \norm{X}_{\CM}^2
        =
        \sum_{j=1}^\infty
        \frac{\lambda_j Z_j^2}{\lambda_j}
        =
        \sum_{j=1}^\infty Z_j^2.
\end{equation}
The last series diverges almost surely.  Therefore
\begin{equation}
        \mu(\mathcal{H}_C)=0,
        \qquad
        \mu(B_r^{\CM})=0
        \quad \text{for every finite } r.
        \label{eq:CM-zero}
\end{equation}

This is the main infinite-dimensional obstruction.  The covariance makes samples
smoother than white noise, but the Cameron--Martin space is smoother still.  The
Cameron--Martin norm removes exactly the covariance damping from every mode.
In finite dimensions this is harmless because there are only finitely many modes:
\(\sum_{j=1}^n Z_j^2<\infty\).  In infinite dimensions, whitening produces
infinitely many independent unit-variance coordinates, and the total whitened
energy diverges.

\subsection{The \texorpdfstring{\(\mathbb{Q}^3\subset\mathbb{R}^3\)}{Q3 in R3} analogy}

It is tempting to imagine \(\mathcal{H}_C\) as a lower-dimensional plane inside
\(H\), like \(\R^2\subset\R^3\).  This is not the best picture.  In many important
cases \(\mathcal{H}_C\) is dense in the ambient space \(H\).  A better analogy is
\begin{equation}
        \mathbb{Q}^3 \subset \mathbb{R}^3.
\end{equation}
The rational points are dense: every open ball in \(\mathbb{R}^3\) contains rational
points.  But \(\mathbb{Q}^3\) has Lebesgue measure zero.  Similarly, the
Cameron--Martin space may be topologically large and geometrically ubiquitous,
yet it is negligible for the Gaussian measure.  Gaussian samples can be approximated
arbitrarily well in the ambient topology by Cameron--Martin elements, but almost
no sample is itself a Cameron--Martin element.

\begin{example}[Brownian motion]
For Wiener measure on \(C[0,1]\), the Cameron--Martin space is
\begin{equation}
        H^1_0[0,1]
        =
        \{h:h(0)=0,\ h'\in L^2[0,1]\}.
\end{equation}
Brownian paths are almost surely nowhere differentiable, so they are not in
\(H^1_0\).  Nevertheless, \(H^1_0\) is the space of finite-energy translations of
Wiener measure.  It is geometrically central even though it has probability zero.
\end{example}

\subsection{Interpretation}

The Gaussian prior contains two distinct structures:
\begin{enumerate}[label=(\roman*)]
\item a probability measure on the ambient model space, describing typical random
samples;
\item a Cameron--Martin geometry, describing finite-energy deterministic
perturbations and covariance-adapted smoothness.
\end{enumerate}
These coincide in finite dimensions, but they separate in function spaces.  This is
why Cameron--Martin balls are excellent deterministic admissible sets but generally
not credible regions of positive Gaussian probability.

\section{Hardening Gaussian priors in function spaces}

The preceding section shows that the exact Cameron--Martin ball cannot be
probability-calibrated in infinite dimensions.  This does not mean there is no good
Gaussian hardening.  It means that one must choose a gauge weaker than the full
Cameron--Martin norm, or work with finite-dimensional projections.

\subsection{Quadratic gauges}

Let \(A:H\to H\) be a positive self-adjoint operator.  Define
\begin{equation}
        R_A(m)^2
        =
        \ip{A(m-m_0)}{m-m_0}_H.
\end{equation}
The associated hard prior set is
\begin{equation}
        (\Smodelzero)_A(r)
        =
        \{m:R_A(m)^2\le r^2\}.
\end{equation}
This set has positive and finite Gaussian mass provided the random variable
\(R_A(X)^2\) is finite with positive probability.  A convenient sufficient condition
is that
\begin{equation}
        A^{1/2} C A^{1/2}
        \quad\text{is trace-class.}
        \label{eq:trace_condition}
\end{equation}
If \(A\) and \(C\) commute and share eigenfunctions, with \(Ae_j=a_j e_j\), then
\begin{equation}
        R_A(X)^2
        =
        \sum_{j=1}^\infty a_j\lambda_j Z_j^2.
\end{equation}
Thus the probability-calibrated radius is determined by
\begin{equation}
        \Prob\left(
        \sum_{j=1}^\infty a_j\lambda_j Z_j^2
        \le r_p^2
        \right)=p.
        \label{eq:weightedchisq_general}
\end{equation}
The case \(A=C^{-1}\) gives \(a_j\lambda_j=1\), hence the divergent series
\(\sum_j Z_j^2\).  This is the Cameron--Martin obstruction.

\subsection{A covariance-weighted scale}

A simple family of covariance-shaped hardenings is obtained by taking
\begin{equation}
        A=C^{-\theta},
        \qquad 0\le \theta\le 1.
\end{equation}
Then
\begin{equation}
        R_\theta(m)^2
        =
        \norm{C^{-\theta/2}(m-m_0)}_H^2
        =
        \sum_{j=1}^\infty \lambda_j^{-\theta} x_j^2,
        \qquad
        m-m_0=\sum_j x_j e_j.
\end{equation}
For a Gaussian draw,
\begin{equation}
        R_\theta(X)^2
        =
        \sum_{j=1}^\infty \lambda_j^{1-\theta} Z_j^2.
        \label{eq:theta_weighted_chisq}
\end{equation}
Hence \(R_\theta(X)<\infty\) almost surely when
\begin{equation}
        \sum_{j=1}^\infty \lambda_j^{1-\theta}<\infty.
\end{equation}
The endpoints have clear meanings:
\begin{align}
        \theta=0 &:\quad R_0(m)=\norm{m-m_0}_H, \\
        \theta=1 &:\quad R_1(m)=\norm{m-m_0}_{\CM}.
\end{align}
Thus \(0<\theta<1\) gives intermediate ellipsoids: they retain covariance-adapted
anisotropy but are thicker than the Cameron--Martin ball in high-frequency
directions.  These are often the most useful infinite-dimensional analogues of
finite-dimensional Mahalanobis ellipsoids.

\begin{principle}[Weakening the Cameron--Martin norm]
In function spaces, the exact Cameron--Martin ball follows the covariance contours
but has zero Gaussian mass.  A weakened covariance norm such as
\(\norm{C^{-\theta/2}(m-m_0)}_H\), with \(0<\theta<1\), preserves part of the
covariance geometry while allowing positive Gaussian probability.
\end{principle}

\section{Example: an elliptic smoothing covariance on \texorpdfstring{\(L^2\)}{L2}}

Let the ambient model space be
\begin{equation}
        H=L^2(\Omega),
\end{equation}
and let
\begin{equation}
        L = \kappa^2 - \alpha\Delta
\end{equation}
denote a positive self-adjoint elliptic operator with eigenpairs
\begin{equation}
        L e_j = a_j e_j,
        \qquad a_j>0.
\end{equation}
Consider a smoothing covariance
\begin{equation}
        C = L^{s/2},
        \qquad s<0.
\end{equation}
Then
\begin{equation}
        \lambda_j=a_j^{s/2}.
\end{equation}
High-frequency modes have large \(a_j\), hence small \(\lambda_j\).  The prior
therefore suppresses rough directions.

The exact Cameron--Martin norm is
\begin{equation}
        \norm{C^{-1/2}(m-m_0)}_{L^2}^2
        =
        \norm{L^{-s/4}(m-m_0)}_{L^2}^2.
\end{equation}
Since \(s<0\), the operator \(L^{-s/4}\) is a positive derivative-like operator.
It measures regularity.  But the corresponding ball has Gaussian probability zero
in infinite dimensions.

A weakened covariance ellipsoid is
\begin{equation}
        (\Smodelzero)_{\theta}(r)
        =
        \left\{m:
        \norm{C^{-\theta/2}(m-m_0)}_{L^2}\le r
        \right\}
        =
        \left\{m:
        \norm{L^{-\theta s/4}(m-m_0)}_{L^2}\le r
        \right\},
        \label{eq:elliptic_theta_set}
\end{equation}
where \(0<\theta<1\).  The radius \(r=r_p\) is chosen by
\begin{equation}
        \Prob\left(
        \sum_{j=1}^\infty
        \lambda_j^{1-\theta} Z_j^2
        \le r_p^2
        \right)=p,
\end{equation}
or equivalently
\begin{equation}
        \Prob\left(
        \sum_{j=1}^\infty
        a_j^{s(1-\theta)/2} Z_j^2
        \le r_p^2
        \right)=p.
        \label{eq:elliptic_quantile}
\end{equation}
In practice, this is computed after spectral truncation.

\section{Computing radii in practice}

Suppose a hardening gauge produces
\begin{equation}
        R(X)^2 = \sum_{j=1}^\infty w_j Z_j^2,
        \qquad w_j\ge0,
        \qquad \sum_j w_j<\infty.
\end{equation}
Then the probability-calibrated radius is
\begin{equation}
        r_p^2 = F^{-1}(p),
        \qquad
        F(t)=\Prob\left(\sum_{j=1}^\infty w_j Z_j^2\le t\right).
\end{equation}
There are several practical routes.

\begin{recipe}[Spectral truncation]
Choose \(N\) so that the tail
\begin{equation}
        T_N = \sum_{j>N} w_j
\end{equation}
is negligible relative to the desired accuracy.  Approximate
\begin{equation}
        R(X)^2 \approx R_N(X)^2
        :=\sum_{j=1}^N w_j Z_j^2.
\end{equation}
Compute the \(p\)-quantile of this weighted chi-square sum using Davies' method,
Imhof's method, saddlepoint approximation, or Monte Carlo.  Then set
\(r_p=\sqrt{F_N^{-1}(p)}\).
\end{recipe}

\begin{recipe}[Moment matching]
For a quick approximation, use
\begin{equation}
        \mu_1=\E R_N^2=\sum_{j=1}^N w_j,
        \qquad
        \sigma^2=\operatorname{Var}(R_N^2)=2\sum_{j=1}^N w_j^2.
\end{equation}
Approximate
\begin{equation}
        R_N^2 \approx a\chi^2_\nu,
        \qquad
        a=\frac{\sigma^2}{2\mu_1},
        \qquad
        \nu=\frac{2\mu_1^2}{\sigma^2}.
\end{equation}
Then
\begin{equation}
        r_p^2\approx a\chi^2_\nu(p).
\end{equation}
\end{recipe}

\begin{recipe}[Monte Carlo]
Generate independent samples
\begin{equation}
        X^{(i)}=m_0+\sum_{j=1}^N\sqrt{\lambda_j} Z_j^{(i)}e_j,
        \qquad i=1,\dots,M,
\end{equation}
and compute \(R(X^{(i)})^2\).  The empirical \(p\)-quantile of these values estimates
\(r_p^2\).  This requires only a Gaussian sampler and a way to evaluate the chosen
gauge.
\end{recipe}

\section{Hardening likelihood kernels}

The prior is only half of the Bayesian setup.  Suppose the likelihood is generated by
\begin{equation}
        d = G(m)+\eta,
        \qquad \eta\sim \nu,
\end{equation}
where \(G:\modelspace\to\dataspace\) is the forward map and \(\nu\) is a noise
measure on \(\dataspace\).  A natural hardening of the likelihood chooses a noise
set \(\Snoise_\beta\subset\dataspace\) satisfying
\begin{equation}
        \nu(\Snoise_\beta)=\beta.
\end{equation}
Then define the set-likelihood
\begin{equation}
        L_{\mathrm{set}}^\beta(m)=G(m)+\Snoise_\beta.
\end{equation}
Equivalently,
\begin{equation}
        d\in L_{\mathrm{set}}^\beta(m)
        \quad\Longleftrightarrow\quad
        d-G(m)\in \Snoise_\beta.
\end{equation}
If \(\eta\sim\mathcal{N}(0,\Gamma)\) in \(\R^D\), then the canonical finite-dimensional
choice is the Mahalanobis ellipsoid
\begin{equation}
        \Snoise_\beta
        =
        \{e:e^T\Gamma^{-1}e\le \chi^2_D(\beta)\}.
\end{equation}
The resulting deterministic feasible model set is
\begin{equation}
        K_{\mathrm{set}}^{p,\beta}(\dobs)
        =
        \Smodelzero(p)
        \cap
        \{m:\dobs-G(m)\in\Snoise_\beta\}.
        \label{eq:prior_likelihood_hardening}
\end{equation}
This set is calibrated by prior probability \(p\) and noise probability \(\beta\), but
its posterior probability is not generally \(p\beta\).  It is a deterministic feasible
set constructed from probability-calibrated ingredients.

\section{Prior hardening versus posterior hardening}

There are two different operations that should not be confused.

\subsection{Prior-plus-likelihood hardening}

One may harden the prior and likelihood separately:
\begin{equation}
        \mprior \leadsto \Smodelzero(p),
        \qquad
        \nu \leadsto \Snoise_\beta,
\end{equation}
and then form \eqref{eq:prior_likelihood_hardening}.  This construction is useful
when the goal is a DLI or PLI feasible set whose assumptions are explicitly
separated into prior model admissibility and data-error admissibility.

\subsection{Posterior hardening}

Alternatively, after observing \(\dobs\), one may form the Bayesian posterior
\(\mpost\) and then choose a posterior credible set
\begin{equation}
        \mathsf{C}_p(\dobs)
        \quad\text{such that}\quad
        \mpost(\mathsf{C}_p(\dobs))=p.
\end{equation}
This is a posterior-to-set operation.  In finite dimensions, a Gaussian posterior gives
an HPD ellipsoid.  In function spaces, a model-space HPD set again depends on the
chosen reference measure.  However, if the quantity of interest is finite-dimensional,
there is a clean alternative.

\section{Property-space hardening}

Let
\begin{equation}
        T:\modelspace\to\propertyspace\simeq\R^P
\end{equation}
be a finite-dimensional property map.  Instead of constructing a set in the whole
model space, push the posterior forward:
\begin{equation}
        \mu_T^d := T_\#\mpost.
\end{equation}
In linear-Gaussian problems this push-forward is a finite-dimensional Gaussian.
For example, if
\begin{equation}
        m\sim\mathcal{N}(m_0,C_0),
        \qquad
        \dobs=Gm+\eta,
        \qquad
        \eta\sim\mathcal{N}(0,\Gamma),
\end{equation}
with \(G\) and \(T\) linear and continuous, then
\begin{equation}
        z=T(m)\mid \dobs
        \sim
        \mathcal{N}(\bar z_d,\Sigma_{z|d}),
\end{equation}
where formally
\begin{align}
        \bar z_d
        &=
        Tm_0
        +
        T C_0 G^*
        (G C_0 G^*+\Gamma)^{-1}
        (\dobs-Gm_0),\\
        \Sigma_{z|d}
        &=
        T C_0 T^*
        -
        T C_0 G^*
        (G C_0 G^*+\Gamma)^{-1}
        G C_0 T^*.
\end{align}
The property-space HPD set is then the ellipsoid
\begin{equation}
        K_p(\dobs)
        =
        \left\{z\in\R^P:
        (z-\bar z_d)^T\Sigma_{z|d}^{-1}(z-\bar z_d)
        \le \chi^2_P(p)
        \right\}.
\end{equation}
This is often the cleanest bridge between Bayesian and set-theoretic inference:
\begin{equation}
        \text{posterior measure on models}
        \quad\leadsto\quad
        \text{posterior measure on properties}
        \quad\leadsto\quad
        \text{finite-dimensional property set}.
\end{equation}
It avoids the non-canonical problem of model-space density level sets in function
spaces.

\section{Connection with Backus: hardening soft bounds and softening hard bounds}

Backus' 1988 paper studies the opposite direction: starting from a hard quadratic
bound
\begin{equation}
        Q_X(x_E,x_E)\le 1
\end{equation}
and trying to replace it by a probability distribution on the model space.  His
conclusion is that this softening is not innocuous.  A hard bound can be softened to
many different probability distributions, but such distributions typically contain much
more information than the original inequality.  In infinite dimensions, natural-looking
softenings can even assign probability one to models for which the original quadratic
quantity is infinite \citep{backus1988hardsoft}.

The present note makes the dual point.  If one starts from a soft Gaussian prior and
hardens it to a deterministic set, one must discard probabilistic information.  The
probability measure contains relative weights within every admissible region, while a
set retains only feasible versus infeasible.  Moreover, the hardening is not unique
because there is no canonical volume in function space.  One must choose a gauge:
ambient norm, Cameron--Martin norm, weakened covariance norm, finite-dimensional
projection, or property-space criterion.

Thus the two operations are asymmetric but philosophically parallel:
\begin{align}
        \text{hard}\to\text{soft}
        &:\quad \text{adds probabilistic structure, often unsupported},\\
        \text{soft}\to\text{hard}
        &:\quad \text{discards probabilistic structure, unless a gauge is specified}.
\end{align}
This is why it is better to describe hardening as a modelling operation rather than as
a canonical transformation.

\section{Connection with deterministic linear inference}

Backus--Gilbert style inference and later deterministic formulations emphasize that
finite data cannot generally determine an infinite-dimensional model.  Backus and
Gilbert showed that finite gross Earth data leave an infinite-dimensional set of
acceptable models unless further assumptions are imposed \citep{backusgilbert1967,backusgilbert1968}.
Backus later showed that, except in special cases where prediction functionals are
linear combinations of data functionals, prior information is necessary to obtain useful
bounds on predictions \citep{backus1970i,backus1988hardsoft}.

Modern deterministic linear inference takes this seriously: the aim is not necessarily
to recover the full model but to bound finite-dimensional properties of it.  Al-Attar
emphasizes that the model-space topology and the admissible constraint set are part
of the problem formulation, not mere technicalities.  In particular, restricting an
inference problem to a smoother space can be interpreted as imposing a corresponding
constraint set in the original model space \citep{alattar2021linear}.

This is exactly the role of hardening.  A Gaussian prior supplies possible gauges for
constructing deterministic admissible sets:
\begin{itemize}
\item ambient norm balls, which are probability-calibrated but may ignore covariance
anisotropy;
\item Cameron--Martin balls, which encode covariance geometry but have zero prior
mass;
\item weakened covariance ellipsoids, which retain covariance geometry while having
positive mass;
\item finite-dimensional KL ellipsoids, which recover the finite-dimensional Gaussian
HPD picture;
\item property-space posterior ellipsoids, which are canonical when the property is
finite-dimensional.
\end{itemize}

\section{Summary of choices}

\begin{center}
\begin{tabular}{lll}
\toprule
Set type & Probability mass? & Interpretation \\
\midrule
Ambient ball \(\norm{m-m_0}_H\le r\) & positive, calibratable & size/energy in ambient space \\
CM ball \(\norm{C^{-1/2}(m-m_0)}_H\le r\) & usually zero & hard covariance-energy constraint \\
Fractional ball \(\norm{C^{-\theta/2}(m-m_0)}_H\le r\) & positive if trace condition holds & weakened covariance geometry \\
KL-truncated CM ellipsoid & positive & finite-dimensional Mahalanobis set \\
Posterior property ellipsoid & positive & finite-dimensional inference set \\
\bottomrule
\end{tabular}
\end{center}

The most important practical recommendation is:
\begin{quote}
If the goal is a model-space deterministic prior set that both reflects the covariance
and has prescribed Gaussian mass, use a weakened covariance ellipsoid or a
finite-dimensional KL truncation, not the full Cameron--Martin ball.
\end{quote}
If the goal is a deterministic regularity assumption rather than a credible set, then a
Cameron--Martin ball is perfectly legitimate.  It should simply not be described as
containing prior probability \(p>0\).

\section{Thesis-ready formulation}

A concise version suitable for insertion into a thesis is the following.

\begin{quote}
The passage from a soft prior measure to a hard admissible set is not canonical in
function spaces.  In finite dimensions, a Gaussian prior has a density with respect to
Lebesgue measure, and the highest-density regions are Mahalanobis ellipsoids.
These ellipsoids also minimize Lebesgue volume among sets of fixed Gaussian
probability.  This gives a natural finite-dimensional hardening.

In infinite-dimensional model spaces there is no canonical Lebesgue measure, so
there is no reference-free notion of minimum-volume HPD set.  A Gaussian prior
nevertheless induces a covariance operator and a Cameron--Martin geometry.  The
Cameron--Martin ball is the formal infinite-dimensional analogue of the
Mahalanobis ellipsoid, but it typically has Gaussian measure zero because Gaussian
samples almost surely do not belong to the Cameron--Martin space.  The
Cameron--Martin space should therefore be interpreted as a covariance-induced
finite-energy geometry, not as the typical sample space of the Gaussian measure.

A probability-calibrated hardening must instead choose a gauge whose balls have
positive Gaussian mass.  Examples include ambient norm balls, finite-dimensional
KL ellipsoids, or weakened covariance ellipsoids
\[
        \{m:\norm{C^{-\theta/2}(m-m_0)}\le r_p\},
        \qquad 0<\theta<1,
\]
where \(r_p\) is chosen from the distribution of
\[
        \sum_j \lambda_j^{1-\theta}Z_j^2.
\]
This construction preserves part of the Gaussian covariance geometry while
remaining probability-calibrated.  It is a hardening operation: it retains a selected
probability-calibrated admissible set and discards the posterior or prior weighting
inside that set.
\end{quote}

\appendix

\section{A short proof that Cameron--Martin balls have zero mass}

Let \(X\sim\mathcal{N}(0,C)\) on a separable Hilbert space \(H\), and let
\(Ce_j=\lambda_j e_j\) with \(\lambda_j>0\).  Then
\begin{equation}
        X=\sum_j \sqrt{\lambda_j}Z_j e_j.
\end{equation}
The Cameron--Martin norm is
\begin{equation}
        \norm{h}_{\CM}^2=\sum_j \frac{h_j^2}{\lambda_j}.
\end{equation}
Hence
\begin{equation}
        \norm{X}_{\CM}^2
        =
        \sum_j Z_j^2.
\end{equation}
Since the \(Z_j^2\) are non-negative independent random variables with mean one,
\begin{equation}
        \frac1N\sum_{j=1}^N Z_j^2 \to 1
        \quad \text{almost surely},
\end{equation}
by the strong law of large numbers.  Therefore
\begin{equation}
        \sum_{j=1}^\infty Z_j^2=\infty
        \quad\text{almost surely}.
\end{equation}
Thus \(X\notin\mathcal{H}_C\) almost surely, and every finite Cameron--Martin
ball has Gaussian probability zero.

\section{Reference-measure dependence of density level sets}

If \(\mu\) is a probability measure and \(\nu\) is a reference measure with
\(\mu\ll\nu\), then a density level set has the form
\begin{equation}
        \left\{m:\frac{\dd\mu}{\dd\nu}(m)\ge c\right\}.
\end{equation}
Changing \(\nu\) changes the density and hence changes its level sets.  In finite
dimensions Lebesgue measure is usually treated as canonical.  In function spaces
there is no such canonical reference measure.  Therefore model-space HPD sets are
not intrinsic unless a reference measure is specified.

In Bayesian inverse problems on function spaces, one often writes the posterior
relative to the prior:
\begin{equation}
        \frac{\dd\mpost}{\dd\mprior}(m)
        \propto
        \exp[-\Phi(m;d)].
\end{equation}
The resulting density level sets are likelihood or data-misfit level sets, not full
prior-plus-likelihood Mahalanobis ellipsoids.  If one writes both prior and posterior
relative to another Gaussian reference measure, the level sets change.  This is not a
paradox; it is the ordinary reference-measure dependence of densities.

\begin{thebibliography}{99}

\bibitem[Al-Attar(2021)]{alattar2021linear}
Al-Attar, D. (2021).
\newblock Linear inference problems with deterministic constraints.
\newblock Manuscript/arXiv version.

\bibitem[Backus(1970a)]{backus1970i}
Backus, G. (1970a).
\newblock Inference from inadequate and inaccurate data, I.
\newblock \emph{Proceedings of the National Academy of Sciences}, 65, 1--7.

\bibitem[Backus(1970b)]{backus1970ii}
Backus, G. (1970b).
\newblock Inference from inadequate and inaccurate data, II.
\newblock \emph{Proceedings of the National Academy of Sciences}, 65, 281--287.

\bibitem[Backus(1970c)]{backus1970iii}
Backus, G. (1970c).
\newblock Inference from inadequate and inaccurate data, III.
\newblock \emph{Proceedings of the National Academy of Sciences}, 67, 282--289.

\bibitem[Backus(1988)]{backus1988hardsoft}
Backus, G. E. (1988).
\newblock Comparing hard and soft prior bounds in geophysical inverse problems.
\newblock \emph{Geophysical Journal}, 94, 249--261.

\bibitem[Backus(1989)]{backus1989csi}
Backus, G. E. (1989).
\newblock Confidence set inference with a prior quadratic bound.
\newblock \emph{Geophysical Journal}, 97, 119--150.

\bibitem[Backus and Gilbert(1967)]{backusgilbert1967}
Backus, G. E. and Gilbert, J. F. (1967).
\newblock Numerical applications of a formalism for geophysical inverse problems.
\newblock \emph{Geophysical Journal of the Royal Astronomical Society}, 13, 247--276.

\bibitem[Backus and Gilbert(1968)]{backusgilbert1968}
Backus, G. E. and Gilbert, F. (1968).
\newblock The resolving power of gross Earth data.
\newblock \emph{Geophysical Journal of the Royal Astronomical Society}, 16, 169--205.

\bibitem[Bogachev(1996)]{bogachev1996gaussian}
Bogachev, V. I. (1996).
\newblock Gaussian measures on linear spaces.
\newblock \emph{Journal of Mathematical Sciences}, 79, 933--1047.

\bibitem[Dashti and Stuart(2013)]{dashti2013bayesian}
Dashti, M. and Stuart, A. M. (2013).
\newblock The Bayesian approach to inverse problems.
\newblock In \emph{Handbook of Uncertainty Quantification} / lecture notes and related versions.

\bibitem[Evans and Stark(2002)]{evansstark2002}
Evans, S. N. and Stark, P. B. (2002).
\newblock Inverse problems as statistics.
\newblock \emph{Inverse Problems}, 18, R55--R97.

\bibitem[Kallenberg(1997)]{kallenberg1997foundations}
Kallenberg, O. (1997).
\newblock \emph{Foundations of Modern Probability}.
\newblock Springer.

\bibitem[Parker(1977)]{parker1977linear}
Parker, R. L. (1977).
\newblock Linear inference and underparameterized models.
\newblock \emph{Reviews of Geophysics and Space Physics}, 15, 446--456.

\bibitem[Stuart(2010)]{stuart2010inverse}
Stuart, A. M. (2010).
\newblock Inverse problems: A Bayesian perspective.
\newblock \emph{Acta Numerica}, 19, 451--559.

\end{thebibliography}

\end{document}
