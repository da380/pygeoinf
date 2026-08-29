import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
import pyshtools.expand as pe
import pygeoinf2 as gi
from pygeoinf2.symmetric_space.sphere import Sobolev
counts={"expand":0,"grid":0}
_e,_g=pe.SHExpandDH,pe.MakeGridDH
pe.SHExpandDH=lambda *a,**k:(counts.__setitem__("expand",counts["expand"]+1),_e(*a,**k))[1]
pe.MakeGridDH=lambda *a,**k:(counts.__setitem__("grid",counts["grid"]+1),_g(*a,**k))[1]
def reset(): counts.update(expand=0,grid=0)
rng=np.random.default_rng(0)
for lmax in (32,128,256):
    S=Sobolev(lmax,2.0,0.2); L=S.with_order(0.0); R2=gi.EuclideanSpace(2)
    Lcod=gi.DirectSum([L,L,L,R2]); Scod=gi.DirectSum([S,S,S,R2])
    # trivial L2 operator on grids: no transforms of its own
    def val(u): g=L.grid_values(u); return (L.from_grid_values(g),L.from_grid_values(2*g),L.from_grid_values(3*g),np.array([g[0,0],g[1,1]]))
    def adj(v):
        v1,v2,v3,a=v; g=L.grid_values(v1)+2*L.grid_values(v2)+3*L.grid_values(v3); g=g.copy(); g[0,0]+=a[0]; g[1,1]+=a[1]; return L.from_grid_values(g)
    A_l2=gi.LinearOperator.from_callables(L,Lcod,val,adjoint=adj)
    A=gi.LinearOperator.from_formal_adjoint(S,Scod,A_l2)
    x=S.random(rng=rng); y=Scod.random(rng=rng)
    reset(); A_l2(x); base_f=dict(counts); reset(); A_l2.adjoint(y); base_a=dict(counts)
    reset(); A(x); lift_f=dict(counts); reset(); A.adjoint(y); lift_a=dict(counts)
    # timings interleaved
    tf=ta=tb=0; n=5
    for _ in range(n):
        t=time.perf_counter(); A_l2.adjoint(y); tb+=time.perf_counter()-t
        t=time.perf_counter(); A.adjoint(y); ta+=time.perf_counter()-t
        t=time.perf_counter(); A(x); tf+=time.perf_counter()-t
    print(f"lmax={lmax}: transforms L2 fwd {base_f} adj {base_a} | lifted fwd {lift_f} adj {lift_a} | time L2 adj {1e3*tb/n:.1f} ms, lifted adj {1e3*ta/n:.1f} ms, lifted fwd {1e3*tf/n:.1f} ms")
