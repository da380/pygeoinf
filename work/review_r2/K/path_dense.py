import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, scipy.sparse as sps
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.symmetric_space import base
rng=np.random.default_rng(1)
X=Sobolev(48,2.0,0.1); rec=X.stations(count=24,rng=rng); src=X.earthquakes(count=40,minimum_magnitude=5.5,rng=rng)
paths=[(s,r) for s in src for r in rec]
t=time.perf_counter(); A=X.path_average_operator(paths,count=16,dense=True); T=time.perf_counter()-t
print(f"path_average_operator(dense=True) total: {T:.2f}s for {len(paths)} paths x16 nodes, dim {X.dim}")
# pieces
t=time.perf_counter(); nodes=[]; rows=[];cols=[];vals=[]
for i,(s,e) in enumerate(paths):
    n,w=X.geodesic_quadrature(s,e,count=16); off=len(nodes); nodes+=n; rows+=[i]*16; cols+=list(range(off,off+16)); vals+=(w/w.sum()).tolist()
tq=time.perf_counter()-t
t=time.perf_counter(); B=X.basis_matrix(nodes); tb=time.perf_counter()-t
W=sps.coo_matrix((vals,(rows,cols)),shape=(len(paths),len(nodes))).tocsr()
Wop=base._weight_operator(len(paths),len(nodes),rows,cols,vals)
t=time.perf_counter(); Wd=Wop.matrix(form="components"); tw=time.perf_counter()-t
t=time.perf_counter(); M1=Wd@B; td=time.perf_counter()-t
t=time.perf_counter(); M2=W@B; ts=time.perf_counter()-t
print(f"quadrature nodes {tq:.2f}s | basis_matrix {tb:.2f}s | weights.matrix() extraction {tw:.2f}s | dense W@B {td:.2f}s | sparse W@B {ts:.3f}s | agree {np.abs(M1-M2).max():.1e}")
# vectorised quadrature prototype
t=time.perf_counter()
P=np.array([X._to_vector(p) for p in [q for pr in paths for q in pr]]).reshape(len(paths),2,3)
a,b=P[:,0],P[:,1]; ang=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))
x,w=np.polynomial.legendre.leggauss(16); s=0.5*(x+1)
N=(np.sin((1-s)[None,:,None]*ang[:,None,None])*a[:,None,:]+np.sin(s[None,:,None]*ang[:,None,None])*b[:,None,:])/np.sin(ang)[:,None,None]
N=N.reshape(-1,3); lat=np.degrees(np.arcsin(N[:,2]/np.linalg.norm(N,axis=1))); lon=np.degrees(np.arctan2(N[:,1],N[:,0]))
tv=time.perf_counter()-t
print(f"vectorised node generation {tv:.3f}s; max node discrepancy {np.abs(np.array(nodes)[:,0]-lat).max():.1e} deg lat")
