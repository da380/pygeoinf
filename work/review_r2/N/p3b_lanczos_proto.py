"""Prototype: Lanczos on a DiagonalMetricSpace keeping the basis in components (one transform per step)."""
from common import *
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.traits import Traits
from pygeoinf2.numerics.functional_calculus import apply_operator_function
from scipy.linalg import eigh_tridiagonal
rng = np.random.default_rng(5)
cnt = Counts(); cnt.patch_sh()

def proto_apply(op, f, x, k):
    sp = op.domain; g = sp.metric_values
    cx = sp.to_components(x); nrm = np.sqrt(cx @ (g*cx))
    Q = [cx / nrm]; a = []; b = []
    prev_beta = 0.0
    for step in range(k):
        w = sp.to_components(op(sp.from_components(Q[-1])))   # 1 synth + 1 analysis (+ the operator's own)
        alpha = w @ (g*Q[-1]); a.append(alpha)
        w -= alpha*Q[-1]
        if step > 0: w -= prev_beta*Q[-2]
        for q in Q: w -= (w @ (g*q))*q
        beta = np.sqrt(w @ (g*w))
        if beta <= 1e-12*max(abs(alpha), 1.0) or step+1 == k: break
        b.append(beta); prev_beta = beta; Q.append(w/beta)
    vals, vecs = eigh_tridiagonal(np.array(a), np.array(b))
    wts = vecs @ (f(vals)*vecs[0, :])
    return sp.from_components(nrm * (np.stack(Q, axis=1) @ wts))

for lmax in (64, 128):
    sp = Sobolev(lmax, 2.0, 0.2)
    pts = sp.random_points(200, rng=rng)
    P = sp.point_evaluation_operator(pts)
    A = (LinearOperator.identity(sp) + P.adjoint @ P).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
    x = sp.random(rng=rng)
    for rep in range(2):
        cnt.reset(); t = time.perf_counter(); y1 = apply_operator_function(A, np.sqrt, x, max_iterations=30, rtol=0.0); t1 = time.perf_counter()-t; c1 = dict(cnt.c)
        cnt.reset(); t = time.perf_counter(); y2 = proto_apply(A, np.sqrt, x, 30); t2 = time.perf_counter()-t; c2 = dict(cnt.c)
        print(f"lmax {lmax} rep {rep}: v2 {t1:.2f}s {c1}  proto {t2:.2f}s {c2}  ratio {t1/t2:.1f}  diff {sp.norm(sp.subtract(y1,y2))/sp.norm(y1):.1e}")
