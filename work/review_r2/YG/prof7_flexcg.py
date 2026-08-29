"""Is the lifted flexure operator self-adjoint on a Sobolev space? Does CG on it give the inverse?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import TransformCounter
import numpy as np
from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2, Lebesgue as Leb2
from pygeoinf2.symmetric_space.circle import Sobolev as CSob
from pygeoinf2.testing import check_traits
from pygeoinf2.traits import Traits
from pygeoinf2.numerics.solvers import CGSolver
import pygeoinf.symmetric_space.sphere as sph1

rng = np.random.default_rng(7)
for name, X in (("sphere lmax 24, H^2", Sob2(24, 2.0, 0.2)), ("circle 64, H^2", CSob(64, 2.0, 0.1))):
    L2 = X.with_order(0.0)
    D = L2.add(L2.project_function(lambda p: 2.0), L2.heat_measure(0.3).sample(rng=rng))
    F = X.flexural_operator(D, 0.25, 1.0)
    x = X.heat_measure(0.3).sample(rng=rng); y = X.heat_measure(0.3).sample(rng=rng)
    lhs = X.inner_product(F(x), y); rhs = X.inner_product(x, F.adjoint(y)); sym = X.inner_product(x, F(y))
    print(f"{name}: (Fx,y)={lhs:.6g} (x,F*y)={rhs:.6g} (x,Fy)={sym:.6g}  -> self-adjoint in H^s? rel diff {abs(lhs-sym)/abs(lhs):.2e}")
    Finv = X.inverse_flexural_operator(D, 0.25, 1.0)
    try:
        z = Finv(y)
        r = X.subtract(F(z), y)
        print(f"   inverse_flexural: ||F Finv y - y||/||y|| = {X.norm(r)/X.norm(y):.2e}")
    except Exception as e:
        print(f"   inverse_flexural FAILED: {str(e)[:120]}")
        z = None
    # the fix: lift the L2 inverse
    from pygeoinf2.symmetric_space.base import lift_formal_adjoint
    Fix = lift_formal_adjoint(L2.inverse_flexural_operator(D, 0.25, 1.0), X)
    zf = Fix(y)
    print(f"   lifted-L2-inverse residual in H^s: {X.norm(X.subtract(F(zf), y))/X.norm(y):.2e}")
    a = X.inner_product(Fix(x), y); b = X.inner_product(x, Fix.adjoint(y))
    print(f"   lifted inverse adjoint identity: {a:.9g} vs {b:.9g} (CG rtol 1e-8)")
    print(f"   traits claimed on the operator CG sees: {F.with_traits(Traits.POSITIVE_DEFINITE).traits!s}")
    try:
        check_traits(F.with_traits(Traits.POSITIVE_DEFINITE), rng=rng)
        print("   check_traits: passed")
    except Exception as e:
        print(f"   check_traits: FAILED: {str(e)[:160]}")
    # the L2 route: solve on L2 with the same CG, compare
    FL = L2.flexural_operator(D, 0.25, 1.0)
    FLinv = L2.inverse_flexural_operator(D, 0.25, 1.0)
    zL = FLinv(L2.from_grid_values(X.grid_values(y)))
    print(f"   L2 route residual: {L2.norm(L2.subtract(FL(zL), L2.from_grid_values(X.grid_values(y))))/L2.norm(zL):.2e}")
    # count CG cost
    with TransformCounter() as c:
        FLinv(L2.from_grid_values(X.grid_values(y)))
    print(f"   transforms per inverse_flexural apply (L2): {c}")

# v1 for comparison
X1 = sph1.Sobolev(24, 2.0, 0.2); L21 = sph1.Lebesgue(24)
D1 = L21.project_function(lambda p: 2.0); L21.axpy(1.0, L21.heat_kernel_gaussian_measure(0.3).sample(), D1)
F1 = X1.flexural_operator(D1, 0.25, 1.0)
x1 = X1.heat_kernel_gaussian_measure(0.3).sample(); y1 = X1.heat_kernel_gaussian_measure(0.3).sample()
print(f"v1 sphere: (Fx,y)={X1.inner_product(F1(x1), y1):.6g} (x,Fy)={X1.inner_product(x1, F1(y1)):.6g}")
Finv1 = X1.inverse_flexural_operator(D1, 0.25, 1.0)
try:
    z1 = Finv1(y1)
    print(f"v1 inverse residual: {X1.norm(X1.subtract(F1(z1), y1))/X1.norm(y1):.2e}")
except Exception as e:
    print("v1 inverse FAILED:", str(e)[:200])
