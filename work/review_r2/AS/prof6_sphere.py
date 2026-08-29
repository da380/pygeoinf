from common import *
import pyshtools.expand as E
from pygeoinf2 import LinearOperator, CGSolver, Traits, SpectralPreconditioner
from pygeoinf2.symmetric_space.sphere import Sobolev, Lebesgue
from pygeoinf2.numerics.randomised import random_eig
rng = np.random.default_rng(0)
ana = Counter(); syn = Counter()
_A, _S = E.SHExpandDH, E.MakeGridDH
E.SHExpandDH = ana.wrap(_A); E.MakeGridDH = syn.wrap(_S)
def reset(): ana.n = syn.n = 0
def report(label):
    print(f"{label:55s} analysis {ana.n:4d}  synthesis {syn.n:4d}")
for lmax in (64, 128):
    print("--- lmax", lmax)
    sp = Sobolev(lmax, 2.0, 0.2)
    x = sp.random(rng=rng); y = sp.random(rng=rng)
    t_a = timeit(lambda: sp.to_components(x), repeats=5); t_s = timeit(lambda: sp.from_components(sp.to_components(x)), repeats=5) - t_a
    print(f"dim {sp.dim}; analysis {t_a*1e3:.1f} ms, synthesis {t_s*1e3:.1f} ms")
    reset(); sp.norm(x); report("norm(x)")
    reset(); sp.inner_product(x, y); report("inner_product(x, y)")
    reset(); sp.zero(); report("zero()")
    reset(); sp.copy(x); report("copy(x)")
    reset(); sp.axpy(1.0, x, y); report("axpy")
    # low-rank factor from from_vectors
    vecs = [sp.random(rng=rng) for _ in range(20)]
    Fv = LinearOperator.from_vectors(sp, vecs)
    c = rng.standard_normal(20)
    reset(); Fv(c); report("from_vectors(rank 20) value")
    reset(); Fv.adjoint(x); report("from_vectors(rank 20) adjoint")
    # CG on the sphere with a diagonal operator (model-space shape)
    C = sp.invariant_measure_covariance if hasattr(sp, "invariant_measure_covariance") else None
    from pygeoinf2 import DiagonalLinearOperator
    D = DiagonalLinearOperator(sp, rng.uniform(1, 2, sp.dim))
    b = sp.random(rng=rng)
    reset(); res = CGSolver(rtol=1e-8)(D).solve(b); report(f"CG on sphere, diagonal op, {res.iterations} it")
    print(f"   -> {ana.n/res.iterations:.1f} analyses + {syn.n/res.iterations:.1f} syntheses per iteration")
    # random_eig factor application
    reset(); eig = random_eig(D, rank=20, rng=rng); report("random_eig(rank 20) build")
    reset(); eig(x); report("LowRankEig(rank 20) apply")
E.SHExpandDH = _A; E.MakeGridDH = _S
