"""Transform counts for Lanczos / Gram-Schmidt / from_vectors on a sphere Sobolev space."""
from common import *
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.traits import Traits
from pygeoinf2.numerics.functional_calculus import apply_operator_function, log_determinant, operator_quadratic_form
from pygeoinf2.numerics.randomised import random_eig, random_trace
rng = np.random.default_rng(2)
cnt = Counts(); cnt.patch_sh()
for lmax in (64, 128, 256):
    sp = Sobolev(lmax, 2.0, 0.2)
    x = sp.random(rng=rng); y = sp.random(rng=rng)
    t, _ = timeit(lambda: sp.inner_product(x, y), 5)
    print(f"lmax {lmax} dim {sp.dim}: one inner_product {t*1e3:.2f} ms; axpy {timeit(lambda: sp.axpy(1.0, x, y), 5)[0]*1e3:.3f} ms")
lmax = 64
sp = Sobolev(lmax, 2.0, 0.2)
pts = sp.random_points(200, rng=rng)
P = sp.point_evaluation_operator(pts)
A = (LinearOperator.identity(sp) + P.adjoint @ P).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
cnt.reset(); Ac = counting_operator(A, cnt)
x = sp.random(rng=rng)
t = time.perf_counter(); yv = apply_operator_function(Ac, np.sqrt, x, max_iterations=30, rtol=1e-10); dt = time.perf_counter()-t
print(f"apply_operator_function lmax {lmax}, 30 steps: {dt:.2f}s counts {cnt}")
cnt.reset(); t = time.perf_counter(); _ = [A(x) for _ in range(30)]; dt2 = time.perf_counter()-t
print(f"  30 bare applications: {dt2:.2f}s counts {cnt}  -> reorth/bookkeeping share {(dt-dt2)/dt:.0%}")
cnt.reset(); t = time.perf_counter(); q = operator_quadratic_form(Ac, np.log, x, max_iterations=30, rtol=1e-10); dt = time.perf_counter()-t
print(f"operator_quadratic_form 30 steps: {dt:.2f}s counts {cnt}")
# Gram-Schmidt on 50 vectors
vecs = [sp.random(rng=rng) for _ in range(50)]
cnt.reset(); t = time.perf_counter(); basis = sp.orthonormal_basis(vecs); dt = time.perf_counter()-t
print(f"orthonormal_basis(50) lmax {lmax}: {dt:.2f}s counts {cnt}")
# prototype: components once, weighted MGS in components, synthesise once
def proto_orth(sp, vecs):
    C = np.stack([sp.to_components(v) for v in vecs], axis=1)
    g = sp.metric_values
    Q = []
    for j in range(C.shape[1]):
        v = C[:, j].copy()
        for _ in range(2):
            for q in Q: v -= (q @ (g * v)) * q
        nv = np.sqrt(v @ (g * v)); Q.append(v / nv)
    return [sp.from_components(q) for q in Q]
cnt.reset(); t = time.perf_counter(); basis2 = proto_orth(sp, vecs); dt = time.perf_counter()-t
print(f"  prototype component MGS: {dt:.2f}s counts {cnt}")
# from_vectors adjoint
F = LinearOperator.from_vectors(sp, basis, orthonormal=True)
cnt.reset(); t = time.perf_counter(); c = F.adjoint(x); dt = time.perf_counter()-t
print(f"from_vectors(50).adjoint: {dt:.3f}s counts {cnt}")
# random_eig rank 20 on A
cnt.reset(); t = time.perf_counter(); e = random_eig(Ac, rank=20, rng=rng); dt = time.perf_counter()-t
print(f"random_eig rank 20 lmax {lmax}: {dt:.2f}s counts {cnt}")
# log_determinant stochastic, 20 probes
cnt.reset(); t = time.perf_counter(); ld = log_determinant(A, method="stochastic", samples=10, rng=rng, max_iterations=30); dt = time.perf_counter()-t
print(f"log_determinant stochastic 10 probes: {dt:.2f}s counts {cnt} -> {ld}")
