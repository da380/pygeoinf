from common import *
import pyshtools.expand as E
from pygeoinf2 import LinearOperator, CGSolver, Traits, DiagonalLinearOperator, BlockLinearOperator, ColumnLinearOperator
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.numerics.randomised import random_eig
rng = np.random.default_rng(0)
ana = Counter(); syn = Counter()
_A, _S = E.SHExpandDH, E.MakeGridDH
E.SHExpandDH = ana.wrap(_A); E.MakeGridDH = syn.wrap(_S)
def reset(): ana.n = syn.n = 0
def report(label): print(f"{label:60s} analysis {ana.n:5d}  synthesis {syn.n:4d}")
lmax = 128
sp = Sobolev(lmax, 2.0, 0.2)
x = sp.random(rng=rng)
# block operators on spheres
I = LinearOperator.identity(sp); O = LinearOperator.zero(sp)
B = BlockLinearOperator([[I, O], [I, I]])
reset(); B((x, x)); report("Block [[I,0],[I,I]] on two spheres: apply")
C = ColumnLinearOperator([I, I])
reset(); C.adjoint((x, x)); report("Column [I; I] adjoint on sphere")
# orthonormal_basis cost
vecs = [sp.random(rng=rng) for _ in range(30)]
reset(); t = timeit(lambda: sp.orthonormal_basis(vecs), repeats=1); report(f"orthonormal_basis(30 vectors) {t*1e3:.0f} ms")
def ob_components(vectors):
    C = np.stack([sp.to_components(v) for v in vectors])   # k analyses
    g = sp.metric_values
    out = []
    for c in C:
        orig = np.sqrt(np.dot(c, g*c)); v = c.copy()
        for w in out: v -= np.dot(v, g*w) * w
        nrm = np.sqrt(np.dot(v, g*v))
        if nrm < orig/np.sqrt(2):
            for w in out: v -= np.dot(v, g*w) * w
            nrm = np.sqrt(np.dot(v, g*v))
        if nrm > 1e-10*orig: out.append(v/nrm)
    return [sp.from_components(v) for v in out]           # k syntheses
reset(); t2 = timeit(lambda: ob_components(vecs), repeats=1); report(f"  prototype in components {t2*1e3:.0f} ms")
a = sp.orthonormal_basis(vecs); b = ob_components(vecs)
print("  agree:", max(sp.norm(sp.subtract(u, v)) for u, v in zip(a, b)) < 1e-8)
# random_eig on the sphere
D = DiagonalLinearOperator(sp, rng.uniform(1, 2, sp.dim))
reset(); t = timeit(lambda: random_eig(D, rank=20, rng=rng), repeats=1); report(f"random_eig(D, rank 20) on sphere {t*1e3:.0f} ms")
# --- CG: on the sphere vs in coefficient space ---------------------------------
from conftest import WeightedSpace
W = WeightedSpace(sp.metric_values)
f = sp.from_components(np.abs(sp.to_components(sp.random(rng=rng))) * 0 + 1.0)  # placeholder
# a grid-acting SPD operator: u -> 2u + (m * u) with m a smooth positive field, lifted L2-style
mfield = sp.random(rng=rng); mv = sp.grid_values(mfield); mv = 1.0 + (mv - mv.min())/(mv.max()-mv.min())
mgrid = sp.from_grid_values(mv)
Mul = sp.multiplication_operator(mgrid)
print("multiplication_operator traits:", Mul.traits, type(Mul).__name__)
Agrid = (Mul.adjoint @ Mul + LinearOperator.identity(sp)).with_traits(Traits.POSITIVE_DEFINITE)
b = sp.random(rng=rng)
reset(); res = CGSolver(rtol=1e-6)(Agrid).solve(b); report(f"CG on sphere, grid operator M*M+I, {res.iterations} it")
print(f"   -> {ana.n/res.iterations:.1f} analyses + {syn.n/res.iterations:.1f} syntheses per iteration")
Ac = LinearOperator.from_callables(W, W, lambda c: sp.to_components(Agrid(sp.from_components(c))), adjoint=lambda c: sp.to_components(Agrid(sp.from_components(c))), traits=Traits.POSITIVE_DEFINITE)
bc = sp.to_components(b)
reset(); resc = CGSolver(rtol=1e-6)(Ac).solve(bc); report(f"CG in coefficients, same operator, {resc.iterations} it")
print(f"   -> {ana.n/resc.iterations:.1f} analyses + {syn.n/resc.iterations:.1f} syntheses per iteration")
print("   solutions agree:", sp.norm(sp.subtract(res.solution, sp.from_components(resc.solution)))/sp.norm(res.solution))
inv_s = CGSolver(rtol=1e-6)(Agrid); inv_c = CGSolver(rtol=1e-6)(Ac)
r = interleave({"CG on sphere": lambda: inv_s.solve(b), "CG in coefficients": lambda: inv_c.solve(bc)}, repeats=3)
print({k: f"{v*1e3:.0f} ms" for k, v in r.items()})
# diagonal operator case
Dc = DiagonalLinearOperator(W, D.eigenvalues)
inv_sd = CGSolver(rtol=1e-8)(D); inv_cd = CGSolver(rtol=1e-8)(Dc)
r = interleave({"CG on sphere (diag op)": lambda: inv_sd.solve(b), "CG in coefficients (diag op)": lambda: inv_cd.solve(bc)}, repeats=3)
print({k: f"{v*1e3:.1f} ms" for k, v in r.items()})
E.SHExpandDH = _A; E.MakeGridDH = _S
