import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, sys
import pyshtools.expand as pe
import pygeoinf2 as gi
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.numerics.randomised import random_range
counts={"expand":0,"grid":0}
_e,_g=pe.SHExpandDH,pe.MakeGridDH
pe.SHExpandDH=lambda *a,**k:(counts.__setitem__("expand",counts["expand"]+1),_e(*a,**k))[1]
pe.MakeGridDH=lambda *a,**k:(counts.__setitem__("grid",counts["grid"]+1),_g(*a,**k))[1]
rng=np.random.default_rng(0)
lmax=int(sys.argv[1]); rank=int(sys.argv[2])
S=Sobolev(lmax,2.0,0.2); A=S.sobolev_measure(2.0,0.2,pointwise_std=1.0).covariance
kw=dict(rank=rank) if "rank" in random_range.__code__.co_varnames else {}
counts.update(expand=0,grid=0); t=time.perf_counter(); Q=random_range(A,rank=rank,rng=rng); dt=time.perf_counter()-t
print(f"lmax {lmax} rank {rank}: random_range transforms {dict(counts)} time {dt:.2f}s; basis size {len(Q) if hasattr(Q,'__len__') else '?'}")
# component-space alternative: one transform per vector, Cholesky-QR with Gram
vs=[S.random(rng=rng) for _ in range(rank)]
counts.update(expand=0,grid=0); t=time.perf_counter(); Q2=S.orthonormal_basis(vs); t1=time.perf_counter()-t; c1=dict(counts)
counts.update(expand=0,grid=0); t=time.perf_counter()
C=np.stack([S.to_components(v) for v in vs],axis=1); G=np.stack([S.apply_gram(C[:,i]) for i in range(rank)],axis=1)
M=C.T@G; L=np.linalg.cholesky(M); Qc=np.linalg.solve(L,C.T).T; Q3=[S.from_components(Qc[:,i]) for i in range(rank)]
t2=time.perf_counter()-t; c2=dict(counts)
print(f"  orthonormal_basis({rank} vectors): transforms {c1} time {t1:.2f}s | Cholesky-QR in components: transforms {c2} time {t2:.3f}s")
