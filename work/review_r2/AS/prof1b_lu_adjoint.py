from common import *
from pygeoinf2 import LinearOperator, LUSolver, CholeskySolver, Traits
from pygeoinf2.testing import check_operator, check_traits
rng = np.random.default_rng(1)
n = 60
D = dense_space(n)
G = D._gram
# sanity on the fixture's gram
c = rng.standard_normal(n)
print("solve_gram sanity:", np.linalg.norm(G @ D.solve_gram(c) - c)/np.linalg.norm(c))
Md = rng.standard_normal((n, n)) + n*np.eye(n)
Ad = LinearOperator.from_matrix(D, D, Md, form="components")
Aadj_c = np.linalg.solve(G, Md.T @ G)
print("Ad.adjoint.matrix vs G^-1 M^T G:", np.linalg.norm(Ad.adjoint.matrix(form="components") - Aadj_c)/np.linalg.norm(Aadj_c))
y = rng.standard_normal(n)
ref = np.linalg.solve(Aadj_c, y)
invd = LUSolver()(Ad)
a1 = invd.adjoint(y)
a2 = LUSolver()(Ad.adjoint)(y)
print("invd.adjoint(y) vs exact:", np.linalg.norm(a1-ref)/np.linalg.norm(ref))
print("LU(Ad.adjoint)(y) vs exact:", np.linalg.norm(a2-ref)/np.linalg.norm(ref))
try:
    check_operator(invd, rng=rng); print("check_operator(LU inverse, dense metric): ok")
except AssertionError as e: print("check_operator FAILED:", e)
# Cholesky on a genuinely G-self-adjoint operator: galerkin matrix SPD
S = Md @ Md.T + n*np.eye(n)
Ag = LinearOperator.from_matrix(D, D, S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
check_traits(Ag, rng=rng)
invg = CholeskySolver()(Ag)
x = rng.standard_normal(n)
print("chol inverse residual:", D.norm(D.subtract(Ag(invg(x)), x))/D.norm(x))
print("chol adjoint is self:", invg.adjoint is invg)
# LU on galerkin-form stored matrix (form mismatch → _in_form conversion) 
invl = LUSolver()(Ag)
print("LU on galerkin-stored: residual", D.norm(D.subtract(Ag(invl(x)), x))/D.norm(x), " adjoint err", np.linalg.norm(invl.adjoint(x)-invg(x))/np.linalg.norm(x))
