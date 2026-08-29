import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
import pygeoinf as v1
from pygeoinf.symmetric_space.sphere import Sobolev as Sob1
import pygeoinf2 as gi
from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2
rng=np.random.default_rng(0)
for lmax in (128,256):
    # v1
    S1=Sob1(lmax,2.0,0.2); L1=S1.underlying_space; R2a=v1.EuclideanSpace(2)
    C1=v1.HilbertSpaceDirectSum([S1,S1,S1,R2a]); LC1=v1.HilbertSpaceDirectSum([L1,L1,L1,R2a])
    def val1(u): return [u, 2*u, 3*u, np.array([u.data[0,0],u.data[1,1]])]
    def adj1(v):
        g=v[0]+2*v[1]+3*v[2]; g.data[0,0]+=v[3][0]; g.data[1,1]+=v[3][1]; return g
    A1=v1.LinearOperator.from_formal_adjoint(S1,C1,v1.LinearOperator(L1,LC1,val1,adjoint_mapping=adj1))
    x1=S1.random(); y1=C1.random()
    # v2
    S2=Sob2(lmax,2.0,0.2); L2=S2.with_order(0.0); R2b=gi.EuclideanSpace(2)
    C2=gi.DirectSum([S2,S2,S2,R2b]); LC2=gi.DirectSum([L2,L2,L2,R2b])
    def val2(u): g=L2.grid_values(u); return (L2.from_grid_values(g),L2.from_grid_values(2*g),L2.from_grid_values(3*g),np.array([g[0,0],g[1,1]]))
    def adj2(v):
        g=(L2.grid_values(v[0])+2*L2.grid_values(v[1])+3*L2.grid_values(v[2])).copy(); g[0,0]+=v[3][0]; g[1,1]+=v[3][1]; return L2.from_grid_values(g)
    A2=gi.LinearOperator.from_formal_adjoint(S2,C2,gi.LinearOperator.from_callables(L2,LC2,val2,adjoint=adj2))
    x2=S2.random(rng=rng); y2=C2.random(rng=rng)
    t1=t2=f1=f2=0; n=5
    for _ in range(n):
        t=time.perf_counter(); A1.adjoint(y1); t1+=time.perf_counter()-t
        t=time.perf_counter(); A2.adjoint(y2); t2+=time.perf_counter()-t
        t=time.perf_counter(); A1(x1); f1+=time.perf_counter()-t
        t=time.perf_counter(); A2(x2); f2+=time.perf_counter()-t
    print(f"lmax={lmax} (v1 grid {S1.random().data.shape}, v2 grid {S2.grid_shape}): adjoint v1 {1e3*t1/n:.0f} ms, v2 {1e3*t2/n:.0f} ms | forward v1 {1e3*f1/n:.1f} ms, v2 {1e3*f2/n:.1f} ms")
