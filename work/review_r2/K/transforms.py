import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
import pyshtools.expand as pe
import pygeoinf2 as gi
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.numerics.randomised import random_range, random_eig
counts={"expand":0,"grid":0}
_e,_g=pe.SHExpandDH,pe.MakeGridDH
pe.SHExpandDH=lambda *a,**k:(counts.__setitem__("expand",counts["expand"]+1),_e(*a,**k))[1]
pe.MakeGridDH=lambda *a,**k:(counts.__setitem__("grid",counts["grid"]+1),_g(*a,**k))[1]
def reset(): counts.update(expand=0,grid=0)
rng=np.random.default_rng(0)
S=Sobolev(64,2.0,0.2)
m1=S.sobolev_measure(2.0,0.2,pointwise_std=1.0); m2=S.sobolev_measure(2.0,0.3,pointwise_std=1.0)
lam=S.laplacian_eigenvalues if hasattr(S,"laplacian_eigenvalues") else None
corr=np.tile(np.array([[1.0,-0.5],[-0.5,1.0]]),(S.dim,1,1))
cm=S.correlated_measure_from_correlations([m1.covariance.eigenvalues,m2.covariance.eigenvalues],corr)
print("correlated covariance type:", type(cm.covariance).__name__, "factor type:", type(cm.covariance_factor).__name__)
reset(); s=cm.sample(rng=rng); print("correlated sample transforms:", dict(counts))
x=cm.domain.random(rng=rng); reset(); cm.covariance(x); print("correlated covariance application transforms:", dict(counts), "(floor: 2 expand + 2 grid)")
reset(); m1.sample(rng=rng); print("single sobolev_measure sample transforms:", dict(counts), "(floor: 0 expand + 1 grid)")
reset(); m1.covariance(S.random(rng=rng)); print("single covariance application:", dict(counts))
D1=m1.covariance; D2=m2.covariance
print("D1@D2 type:", type(D1@D2).__name__, "| D1+D2 type:", type(D1+D2).__name__, "| 2*D1 type:", type(2*D1).__name__, "| D1.inverse@D1 type:", type(D1.inverse@D1).__name__)
reset(); (D1@D2)(S.random(rng=rng)); print("(D1@D2) application transforms:", dict(counts))
# random_range on a sphere
A=m1.covariance
reset(); t=time.perf_counter(); Q=random_range(A, 30, rng=rng, power_iterations=2) if "power_iterations" in random_range.__code__.co_varnames else random_range(A,30,rng=rng); dt=time.perf_counter()-t
print("random_range(rank 30) transforms:", dict(counts), f"time {dt:.2f}s", "(operator applications ~ 30*(1+2*2)=150 -> floor ~300 transforms)")
import inspect; print("random_range sig:", inspect.signature(random_range))
