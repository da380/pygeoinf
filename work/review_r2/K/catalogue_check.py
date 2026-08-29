import inspect, numpy as np, scipy.sparse as sp
import pygeoinf2 as gi
from pygeoinf2 import numerics
from pygeoinf2.symmetric_space.sphere import Sphere
from pygeoinf2.numerics import functional_calculus as fc, randomised as rd
from pygeoinf2.inference import DualFeasibleProperty
def has(o,n): return hasattr(o,n)
print("L99 MatrixLinearOperator.assembled:", has(gi.MatrixLinearOperator,"assembled"))
X=gi.EuclideanSpace(4); M=sp.random(4,4,density=0.5,random_state=1)+sp.eye(4)
A=gi.LinearOperator.from_matrix(X,X,M.tocsr(),form="components")
print("L100 sparse from_matrix -> type:", type(A).__name__, "stores sparse:", sp.issparse(A.matrix(form="components")) if hasattr(A,'matrix') else None, "assembled type:", type(A.assembled()).__name__ if hasattr(A,'assembled') else None)
print("L108 Functional:", has(gi,"Functional"))
print("L117-120:", has(gi,"BlockLinearOperator"), has(gi,"ColumnLinearOperator"), has(gi,"RowLinearOperator"))
# L152: iterative solver as preconditioner
Y=gi.EuclideanSpace(6); rng=np.random.default_rng(0); R=rng.standard_normal((6,6)); K=R@R.T+6*np.eye(6)
B=gi.LinearOperator.from_matrix(Y,Y,K,form="components",traits=gi.Traits.SELF_ADJOINT|gi.Traits.POSITIVE_DEFINITE)
try:
    s=gi.FlexibleCGSolver(preconditioner=gi.CGSolver(strict=False,maxiter=2))
    x=s(B)(Y.from_components(np.ones(6))); print("L152 iterative preconditioner accepted; resid", np.linalg.norm(K@Y.to_components(x)-1))
except Exception as e: print("L152 FAILED:", type(e).__name__, e)
print("L158 OperatorFunction:", has(fc,"OperatorFunction"), has(numerics,"OperatorFunction"))
print("L169 LowRank*:", [has(rd,n) for n in ("LowRankEig","LowRankSVD","LowRankCholesky")], "L170 random_range:", has(rd,"random_range"))
print("L218 support_values:", inspect.signature(DualFeasibleProperty.support_values) if has(DualFeasibleProperty,"support_values") else "MISSING")
print("L262 ConstrainedLeastSquares:", has(gi,"ConstrainedLeastSquares"))
print("L302 stations:", inspect.signature(Sphere.stations) if has(Sphere,"stations") else "MISSING")
try:
    st=Sphere(8).stations(); print("   stations() ->", type(st).__name__, np.asarray(st).shape)
except Exception as e: print("   stations() FAILED", type(e).__name__, e)
print("L360 scale_inplace:", has(gi.HilbertSpace,"scale_inplace"))
print("L410 as_multivariate_normal:", has(gi.GaussianMeasure,"as_multivariate_normal"))
import pygeoinf2.symmetric_space.base as sb
print("L416 deflated_pointwise_variance:", [ (m, has(o,"deflated_pointwise_variance")) for m,o in (("GaussianMeasure",gi.GaussianMeasure),("SymmetricSpace",sb.SymmetricSpace),("Sphere",Sphere))], "deflated_diagonal:", has(rd,"deflated_diagonal"))
print("L438:", {n:has(Sphere,n) for n in ("lmax","grid_shape","colatitudes","longitudes","grid_axes")})
print("L443 basis_at:", has(Sphere,"basis_at"))
print("L463:", has(gi.GaussianMeasure,"marginal"), has(gi.GaussianMeasure,"cross_covariance"), has(Sphere,"spectral_correlations"))
print("L464 truncation_degree_for:", inspect.signature(Sphere.truncation_degree_for) if has(Sphere,"truncation_degree_for") else "MISSING")
if has(Sphere,"truncation_degree_for"): print("   value(2,0.2,rtol=1e-8):", Sphere.truncation_degree_for(2,0.2,rtol=1e-8))
print("L481 heat_measure:", has(Sphere,"heat_measure"))
print("L498 earthquakes:", has(Sphere,"earthquakes"))
print("L526 Subset.contains:", has(gi.Subset,"contains"))
print("L542 low_rank_surrogate:", has(gi.LinearGaussianInversion,"low_rank_surrogate"))
