"""Correctness checks for the numerics area, on a dense-metric space."""
from common import *
import warnings
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.traits import Traits
from pygeoinf2.numerics.functional_calculus import (log_determinant, operator_function,
    apply_operator_function, operator_quadratic_form, operator_log, lanczos_tridiagonalise)
from pygeoinf2.numerics.randomised import (random_eig, random_svd, random_trace, random_diagonal,
    deflated_diagonal, random_range)
rng = np.random.default_rng(0)
n = 40
space = make_dense_metric_space(n)
G = space.gram_matrix()
# A self-adjoint PD operator: component matrix A_c = G^-1 S with S symmetric PD
R = rng.standard_normal((n, n)); S = R @ R.T + n*np.eye(n)
A = LinearOperator.from_matrix(space, space, S, form="galerkin",
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
Ac = np.linalg.solve(G, S)
# 1. exact eigen-check: generalized eigenproblem S v = lam G v
lam_true = np.sort(np.linalg.eigvalsh(np.linalg.inv(np.linalg.cholesky(G)) @ S @ np.linalg.inv(np.linalg.cholesky(G)).T))[::-1]
eig = random_eig(A, rank=10, rng=rng, power=3)
print("random_eig top-10 rel err:", np.max(np.abs(eig.eigenvalues - lam_true[:10]) / lam_true[:10]))
# isometry check of factor on dense metric
U = np.stack([space.to_components(eig.factor(e)) for e in np.eye(eig.rank)])
print("U^T G U - I:", np.max(np.abs(U @ G @ U.T - np.eye(eig.rank))))
# 2. random_trace estimates tr(A_c)
est = random_trace(A, samples=4000, rng=rng)
print("random_trace:", est, " tr(A_c) =", np.trace(Ac), " tr(G A_c)=", np.trace(S))
# 3. random_diagonal forms
dc = random_diagonal(A, samples=6000, form="components", rng=rng)
dg = random_diagonal(A, samples=6000, form="galerkin", rng=rng)
print("random_diagonal comp rel err:", np.linalg.norm(dc - np.diag(Ac))/np.linalg.norm(np.diag(Ac)),
      " galerkin rel err:", np.linalg.norm(dg - np.diag(S))/np.linalg.norm(np.diag(S)))
# deflated: exact part only (samples small) vs closed form for the low-rank part
low = random_eig(A, rank=n, rng=rng, power=2)  # full rank => deflated exact
dd = deflated_diagonal(A, rank=n, samples=2, form="components", rng=rng)
print("deflated_diagonal (full rank, components) err:", np.max(np.abs(dd - np.diag(Ac))))
dd = deflated_diagonal(A, rank=n, samples=2, form="galerkin", rng=rng)
print("deflated_diagonal (full rank, galerkin) err:", np.max(np.abs(dd - np.diag(S))))
# 4. log_determinant exact for diagonal with_traits
sp2 = EuclideanSpace(200)
D = DiagonalLinearOperator(sp2, rng.uniform(1, 3, 200))
Dt = D.with_traits(Traits.POSITIVE_DEFINITE)
print("with_traits class:", type(Dt).__name__, " logdet:", log_determinant(Dt), " exact:", np.sum(np.log(D.eigenvalues)))
# log_determinant dense vs stochastic on dense metric
ld_dense = log_determinant(A, method="dense")
ld_sto = log_determinant(A, method="stochastic", samples=2000, rng=rng, max_iterations=40)
print("logdet dense:", ld_dense.value, " slogdet(A_c):", np.linalg.slogdet(Ac)[1], " stochastic:", ld_sto)
# 5. operator_function diagonal on dense metric must be refused
try:
    operator_function(DiagonalLinearOperator(space, np.ones(n)*2), np.sqrt)
    print("operator_function diagonal on dense metric: ACCEPTED (bug?)")
except ValueError as e:
    print("operator_function diagonal on dense metric: refused OK")
# 6. Lanczos early stopping: an operator with 5 distinct eigenvalues
V, _ = np.linalg.qr(rng.standard_normal((n, n)))
lam5 = np.repeat([1., 2., 3., 4., 5.], 8)
Sc = V @ np.diag(lam5) @ V.T   # symmetric in Euclidean
sp3 = EuclideanSpace(n)
A5 = LinearOperator.from_matrix(sp3, sp3, Sc, form="galerkin", traits=Traits.SELF_ADJOINT|Traits.POSITIVE_DEFINITE)
cnt = Counts(); A5c = counting_operator(A5, cnt)
x = sp3.random(rng=rng)
y = apply_operator_function(A5c, np.sqrt, x, max_iterations=50)
print("apply_operator_function on 5-eigenvalue op: applications", cnt.c["apply"], " err", np.linalg.norm(y - V @ (np.sqrt(lam5) * (V.T @ x))))
cnt.reset()
q = operator_quadratic_form(A5c, np.log, x, max_iterations=50)
print("operator_quadratic_form applications", cnt.c["apply"], " err", abs(q - x @ (V @ (np.log(lam5) * (V.T @ x)))))
# 7. random_range(rank=k) length
print("random_range(rank=5) length:", len(random_range(A, rank=5, rng=rng)))
# 8. lanczos_tridiagonalise breakdown_tol exposed?
import inspect
print("lanczos_tridiagonalise params:", list(inspect.signature(lanczos_tridiagonalise).parameters))
# 9. SubgradientDescent evaluation count per iteration
from pygeoinf2.numerics.convex import SubgradientDescent, ProximalGradient, SquaredDistance, NormFunctional
from pygeoinf2.algebra.operators import Functional
calls = Counter()
f = SquaredDistance(space, centre=space.random(rng=rng))
class CountF(Functional):
    def __init__(s): super().__init__(space)
    def _value(s, x): calls["value"] += 1; return f(x)
    def _derivative(s, x): calls["deriv"] += 1; return f.derivative(x)
    @property
    def has_subgradient(s): return True
    def subgradient(s, x): calls["sub"] += 1; return f.gradient(x)
res = SubgradientDescent(max_iterations=20, rule="sqrt").minimise(CountF(), space.zero())
print("SubgradientDescent 20 its: value calls", calls["value"], " sub calls", calls["sub"], " reported evaluations", res.evaluations)
calls.clear()
res = ProximalGradient(max_iterations=20).minimise(CountF(), space.zero(), nonsmooth=NormFunctional(space, weight=0.1))
print("ProximalGradient 20 its: value calls", calls["value"], " deriv", calls["deriv"], " reported", res.evaluations, res.iterations)
# 10. LevelBundleMethod bound bracketing on a piecewise-linear convex function
from pygeoinf2.numerics.convex import LevelBundleMethod, ProximalBundleMethod
sp4 = make_dense_metric_space(6)
c0 = sp4.random(rng=rng)
class PL(Functional):
    def __init__(s): super().__init__(sp4)
    def _value(s, x): return sp4.norm(sp4.subtract(x, c0)) + 0.5*sp4.squared_norm(x)
    def _derivative(s, x): raise NotImplementedError
    @property
    def has_subgradient(s): return True
    def subgradient(s, x):
        d = sp4.subtract(x, c0); nm = sp4.norm(d)
        g = sp4.scale(1.0/nm, d) if nm > 0 else sp4.zero()
        return sp4.axpy(1.0, x, g)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r1 = LevelBundleMethod(tolerance=1e-8, iterations=200).minimise(PL(), sp4.zero())
    r2 = ProximalBundleMethod(tolerance=1e-10, iterations=300).minimise(PL(), sp4.zero())
print("LevelBundle:", r1.value, r1.converged, r1.iterations, " Proximal:", r2.value, r2.converged, r2.iterations)
