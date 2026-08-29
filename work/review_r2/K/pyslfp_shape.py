import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import pyshtools.expand as pe
import pygeoinf2 as gi
from pygeoinf2.symmetric_space.sphere import Sphere, Sobolev, Lebesgue
from pygeoinf2.testing import check_operator
rng=np.random.default_rng(1)
counts={"expand":0,"grid":0}
_e, _g = pe.SHExpandDH, pe.MakeGridDH
def E(*a,**k): counts["expand"]+=1; return _e(*a,**k)
def G(*a,**k): counts["grid"]+=1; return _g(*a,**k)
pe.SHExpandDH=E; pe.MakeGridDH=G
def reset(): counts["expand"]=0; counts["grid"]=0
lmax=32
S=Sobolev(lmax,2.0,0.2); L=S.with_order(0.0); R2=gi.EuclideanSpace(2)
print("A. isinstance(S, Lebesgue):", isinstance(S,Lebesgue), "isinstance(L, Lebesgue):", isinstance(L,Lebesgue), "isinstance(S,Sobolev):", isinstance(S,Sobolev), "S.scale is method:", callable(S.scale), "sampling:", S.sampling)
x=S.random(rng=rng); print("   vector type:", type(x).__name__, "grid shape:", S.grid_values(x).shape)
# B. formal adjoint lift onto [Sob,Sob,Sob,R2] codomain from L2 operator
f1=L.grid_values(L.random(rng=rng)); f2=L.grid_values(L.random(rng=rng))
Lcod=gi.DirectSum([L,L,L,R2]); Scod=gi.DirectSum([S,S,S,R2])
def val(u):
    g=L.grid_values(u); a=L.l2_products_operator([f1,f2])(u)
    return (L.from_grid_values(g), L.from_grid_values(2*g), L.from_grid_values(3*g), a)
def adj(v):
    v1,v2,v3,a=v; g=L.grid_values(v1)+2*L.grid_values(v2)+3*L.grid_values(v3)
    return L.add(L.from_grid_values(g), L.l2_products_operator([f1,f2]).adjoint(a))
A_l2=gi.LinearOperator.from_callables(L,Lcod,val,adjoint=adj)
try:
    A=gi.LinearOperator.from_formal_adjoint(S,Scod,A_l2)
    check_operator(A,rng=rng); print("B. lift onto [Sob,Sob,Sob,R2]: OK, check_operator passed")
    reset(); y=A(S.random(rng=rng)); print("   transforms per forward:", dict(counts))
    reset(); z=A.adjoint(Scod.random(rng=rng)); print("   transforms per adjoint:", dict(counts))
except Exception as e: print("B. FAILED:", type(e).__name__, e)
# C. R2 domain lift
def val2(c): return L.from_grid_values(c[0]*f1+c[1]*f2)
def adj2(u): return L.l2_products_operator([f1,f2])(u)
B_l2=gi.LinearOperator.from_callables(R2,L,val2,adjoint=adj2)
try:
    B=gi.LinearOperator.from_formal_adjoint(R2,S,B_l2); check_operator(B,rng=rng); print("C. R2 -> Sob lift: OK")
except Exception as e: print("C. FAILED:", type(e).__name__, e)
# D. conditioned prior is samplable and satisfies constraint
prior=S.sobolev_measure(2.0,0.2,pointwise_std=1.0)
C=S.l2_products_operator([np.ones(S.grid_shape)])
try:
    cond=prior.condition(C, np.zeros(1))
    print("D. condition: can_sample", cond.can_sample, "precision:", cond.precision is not None)
    s=cond.sample(rng=rng); print("   constraint residual on sample:", float(abs(C(s))[0]), "vs unconditioned", float(abs(C(prior.sample(rng=rng)))[0]))
    prob=gi.LinearForwardProblem(S.point_evaluation_operator([(10.,20.),(-30.,100.)]), error=gi.GaussianMeasure.from_standard_deviation(gi.EuclideanSpace(2),0.1))
    m,d=prob.synthetic_model_and_data(cond, rng=rng); post=gi.LinearGaussianInversion(prob,cond,solver=gi.CholeskySolver())(d); print("   posterior on conditioned prior can_sample:", post.can_sample)
except Exception as e: print("D. FAILED:", type(e).__name__, e)
# E. from_product / sum precision
m1=S.sobolev_measure(2.0,0.2,pointwise_std=1.0); m2=S.heat_measure(0.3,pointwise_std=1.0)
P=gi.GaussianMeasure.from_product([m1,m2]); print("E. from_product precision:", P.precision is not None, "can_sample:", P.can_sample)
Q=m1+m2; print("   sum precision:", Q.precision is not None, "can_sample:", Q.can_sample)
n=gi.EuclideanSpace(3); w=gi.GaussianMeasure.from_standard_deviation(n,0.1); corr=gi.GaussianMeasure.from_standard_deviations(n,np.array([1.,2.,3.]))
try:
    pr=gi.LinearForwardProblem(gi.LinearOperator.identity(n), error=w+corr); print("   chi_squared on summed error:", pr.chi_squared(n.zero(), n.from_components(np.ones(3))))
except Exception as e: print("   chi_squared on summed error FAILED:", type(e).__name__, e)
# F. helpers
print("F. from_coefficient_operator:", hasattr(S,"from_coefficient_operator"), "from_standard_deviations:", hasattr(gi.GaussianMeasure,"from_standard_deviations"), "ProgressCallback:", hasattr(gi,"ProgressCallback"))
print("   correlated_measure_from_correlations takes callable?", end=" ")
try:
    cm=S.correlated_measure_from_correlations([m1.covariance.eigenvalues, m1.covariance.eigenvalues], lambda lam: np.array([[1,-0.5],[-0.5,1]])); print("yes")
except Exception as e: print("no:", type(e).__name__)
# G. DirectSum vectors, coordinate_projection
ds=gi.DirectSum([gi.EuclideanSpace(1)]*3); v=ds.random(rng=rng); print("G. DirectSum vector type:", type(v).__name__, "coordinate_projection:", hasattr(ds,"coordinate_projection"), "coordinate_inclusion:", hasattr(ds,"coordinate_inclusion"))
# H. plotting
import pygeoinf2.plotting as pl
try:
    ax,im=pl.plot(S,x,colorbar_kwargs={"orientation":"horizontal","shrink":0.6},gridlines=True,gridlines_kwargs={"lat_interval":30},map_extent=(-20,40,30,70),title="t"); print("H. plot kwargs: OK")
except Exception as e: print("H. plot kwargs FAILED:", type(e).__name__, e)
pts=[(10.,20.),(-30.,100.),(45.,-60.)]
for kw in ({"c":[1,2,3]},{"data":[1,2,3]}):
    try: pl.plot_points(S,pts,**kw); print("   plot_points", list(kw), "OK")
    except Exception as e: print("   plot_points", list(kw), "FAILED:", type(e).__name__, str(e)[:80])
try:
    pl.plot_corner(gi.GaussianMeasure.from_standard_deviations(gi.EuclideanSpace(2),np.array([1.,2.])), title="x"); print("   plot_corner title OK")
except Exception as e: print("   plot_corner FAILED", e)
# I. solve diagnostics through the estimator
prob=gi.LinearForwardProblem(S.point_evaluation_operator([(10.,20.),(-30.,100.),(0.,0.)]), error=gi.GaussianMeasure.from_standard_deviation(gi.EuclideanSpace(3),0.1))
msgs=[]
est=gi.LinearGaussianInversion(prob,prior,solver=gi.CGSolver(callback=gi.ProgressCallback(report=msgs.append)))
m,d=prob.synthetic_model_and_data(prior,rng=rng); post=est(d); print("I. ProgressCallback messages:", len(msgs), msgs[-1] if msgs else None)
print("   est has solve/result:", [n for n in dir(est) if 'solve' in n.lower() or 'result' in n.lower()])
# J. check_operator measures; samples n_jobs; matrix n_jobs; diagonals n_jobs
import inspect
print("J. check_operator params:", list(inspect.signature(check_operator).parameters), "| samples:", 'n_jobs' in inspect.signature(gi.ProbabilityMeasure.samples).parameters, "| matrix:", 'n_jobs' in inspect.signature(gi.LinearOperator.matrix).parameters, "| diagonals:", 'n_jobs' in inspect.signature(gi.LinearOperator.diagonals).parameters)
# K. point convention
op=S.point_evaluation_operator([(90.,0.)]); u=S.from_grid_values(np.ones(S.grid_shape)); print("K. point eval of 1 at north pole (lat=90):", float(op(u)[0]))
print("   two_point_covariance sig:", inspect.signature(gi.GaussianMeasure.two_point_covariance))
# L. multiply / vector_sqrt semantics
print("L. HilbertModule ops on Sphere:", [n for n in ("multiply","sqrt","divide","power") if hasattr(S,n)])
# M. MassWeightedSpace
from pygeoinf2.algebra.spaces import MassWeightedSpace
print("M. MassWeightedSpace exported from gi:", hasattr(gi,"MassWeightedSpace"), "algebra:", hasattr(gi.algebra,"MassWeightedSpace"))
