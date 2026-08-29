from common import *
import scipy.linalg
import pygeoinf2.numerics.solvers as S
from pygeoinf2 import LinearOperator, EuclideanSpace, LUSolver, CholeskySolver, Traits
from pygeoinf2.testing import check_operator

n = 1500
rng = np.random.default_rng(0)
M = rng.standard_normal((n, n)) + n * np.eye(n)
X = EuclideanSpace(n)
A = LinearOperator.from_matrix(X, X, M, form="components")

c = Counter()
orig = S.lu_factor
S.lu_factor = c.wrap(orig)
inv = LUSolver()(A)
print("lu_factor calls at LUSolver(A):", c.n)
S.lu_factor = orig

one = timeit(lambda: scipy.linalg.lu_factor(M))
build = timeit(lambda: LUSolver()(A))
print(f"single lu_factor {one*1e3:.0f} ms; LUSolver(A) build {build*1e3:.0f} ms; ratio {build/one:.2f}")

# correctness of adjoint solve on a dense-metric space
D = dense_space(200)
Md = np.random.default_rng(1).standard_normal((200, 200)) + 200*np.eye(200)
Ad = LinearOperator.from_matrix(D, D, Md, form="components")
invd = LUSolver()(Ad)
y = D.random(rng=rng)
lhs = invd.adjoint(y)
ref = LUSolver()(Ad.adjoint)(y)
print("LU adjoint via trans=1 vs LU(A*): rel err", np.linalg.norm(lhs-ref)/np.linalg.norm(ref))
# Galerkin-form matrix with Cholesky on dense metric: adjoint is self
Sd = Md @ Md.T + 200*np.eye(200)
Ag = LinearOperator.from_matrix(D, D, D._gram @ Sd, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
invg = CholeskySolver()(Ag)
x = D.random(rng=rng)
print("chol inverse residual:", D.norm(D.subtract(Ag(invg(x)), x))/D.norm(x))
check_operator(invd, rng=rng)
print("check_operator(LU inverse on dense metric) ok")
