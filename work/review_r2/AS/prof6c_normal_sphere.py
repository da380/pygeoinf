from common import *
import inspect
import pyshtools.expand as E
from pygeoinf2 import LinearOperator, CGSolver, Traits, DiagonalLinearOperator, EuclideanSpace, SpectralPreconditioner
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.inference import NormalOperator, NormalDiagonalPreconditioner, LinearForwardProblem
from pygeoinf2.probability import GaussianMeasure
rng = np.random.default_rng(0)
ana = Counter(); syn = Counter()
_A, _S = E.SHExpandDH, E.MakeGridDH
E.SHExpandDH = ana.wrap(_A); E.MakeGridDH = syn.wrap(_S)
def reset(): ana.n = syn.n = 0
def report(label): print(f"{label:62s} analysis {ana.n:5d}  synthesis {syn.n:4d}")
lmax, npts = 64, 500
sp = Sobolev(lmax, 2.0, 0.2)
print("sobolev_measure sig:", inspect.signature(sp.sobolev_measure))
pts = sp.random_points(npts, rng=rng)
A = sp.point_evaluation_operator(pts)
Y = A.codomain
prior = sp.sobolev_measure(2.0, 0.2) if len(inspect.signature(sp.sobolev_measure).parameters) >= 2 else sp.sobolev_measure()
Q = prior.covariance
print("prior covariance:", type(Q).__name__)
x = sp.random(rng=rng)
reset(); A(x); report("point evaluation A(x)")
reset(); A.adjoint(np.ones(npts)); report("A.adjoint(d)")
reset(); Q(x); report("Q(x)")
# --- model-space normal operator: Q^-1 + A* A ---
Qinv = prior.precision if hasattr(prior, "precision") and prior.precision is not None else Q.inverse
N = (Qinv + A.adjoint @ A).with_traits(Traits.POSITIVE_DEFINITE)
b = sp.random(rng=rng)
reset(); res = CGSolver(rtol=1e-6, maxiter=400, strict=False)(N).solve(b); report(f"CG on sphere: Q^-1 + A*A, {res.iterations} it")
print(f"   -> {ana.n/res.iterations:.1f} analyses + {syn.n/res.iterations:.1f} syntheses per iteration")
from conftest import WeightedSpace
W = WeightedSpace(sp.metric_values)
Nc = LinearOperator.from_callables(W, W, lambda c: sp.to_components(N(sp.from_components(c))), adjoint=lambda c: sp.to_components(N(sp.from_components(c))), traits=Traits.POSITIVE_DEFINITE)
bc = sp.to_components(b)
reset(); resc = CGSolver(rtol=1e-6, maxiter=400, strict=False)(Nc).solve(bc); report(f"CG in coefficients, same operator, {resc.iterations} it")
print(f"   -> {ana.n/resc.iterations:.1f} analyses + {syn.n/resc.iterations:.1f} syntheses per iteration")
inv_s = CGSolver(rtol=1e-6, maxiter=400, strict=False)(N); inv_c = CGSolver(rtol=1e-6, maxiter=400, strict=False)(Nc)
r = interleave({"CG on sphere": lambda: inv_s.solve(b), "CG in coefficients": lambda: inv_c.solve(bc)}, repeats=3)
print({k: f"{v*1e3:.0f} ms" for k, v in r.items()})
# --- NormalDiagonalPreconditioner on the data-space normal operator ---
err = GaussianMeasure.from_standard_deviation(Y, 0.1) if hasattr(GaussianMeasure, "from_standard_deviation") else None
Nd = NormalOperator(A, prior, error=err)
reset(); t = timeit(lambda: NormalDiagonalPreconditioner()(Nd), repeats=1); report(f"NormalDiagonalPreconditioner build, {npts} data, {t*1e3:.0f} ms")
print(f"   -> {ana.n/npts:.1f} analyses + {syn.n/npts:.1f} syntheses per data index")
# prototype: one analysis per index when Q is diagonal
def proto():
    d = np.zeros(npts); comp = np.zeros(npts)
    for i in range(npts):
        comp[:] = 0; comp[i] = 1.0
        pulled = A.adjoint(Y.from_components(comp))
        c = sp.to_components(pulled)
        d[i] = np.dot(c, sp.apply_gram(Q.eigenvalues * c))
    return d
reset(); t2 = timeit(proto, repeats=1); report(f"  prototype (components) {t2*1e3:.0f} ms")
ref = NormalDiagonalPreconditioner()(Nd)
print("   agree:", np.allclose(proto() + err.covariance.diagonals(form='galerkin')[0], 1.0/ref._solve_fn(Y.from_components(np.ones(npts)), None).solution))
# --- Spectral preconditioner application on the sphere ---
spec = SpectralPreconditioner(rank=20, rng=rng)
reset(); P = spec(N); report("SpectralPreconditioner(rank 20) build on model-space N")
reset(); P(x); report("SpectralPreconditioner apply")
E.SHExpandDH = _A; E.MakeGridDH = _S
