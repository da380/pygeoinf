from common import *
from pygeoinf2 import LinearOperator, EuclideanSpace, DiagonalLinearOperator, JacobiPreconditioner, Traits, CGSolver
from pygeoinf2.inference import TikhonovFamily
from pygeoinf2.numerics.randomised import random_diagonal
n = 2000
rng = np.random.default_rng(0)
X = EuclideanSpace(n)
R = rng.standard_normal((n, n)); S = R @ R.T / n + np.eye(n)
A = LinearOperator.from_matrix(X, X, S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
offs = list(range(-10, 11))
def np_diags():
    out = np.zeros((len(offs), n))
    for i, k in enumerate(offs):
        d = np.diagonal(S, k)
        if k >= 0: out[i, k:k+d.size] = d
        else: out[i, :d.size] = d
    return out
r = interleave({"MatrixLinearOperator.diagonals(21 offsets)": lambda: A.diagonals(offsets=offs), "np.diagonal loop": np_diags, "main diagonal only": lambda: A.diagonals()}, repeats=3)
print({k: f"{v*1e3:.2f} ms" for k, v in r.items()})
assert np.allclose(A.diagonals(offsets=offs), np_diags())

# Jacobi on a sum node: does it fall to dim applications?
c = Counter()
A_counted = LinearOperator.from_callables(X, X, c.wrap(lambda x: S @ x), adjoint=lambda x: S @ x, traits=Traits.POSITIVE_DEFINITE)
Am = LinearOperator.from_matrix(X, X, S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
D = DiagonalLinearOperator(X, np.full(n, 0.5))
for name, op in [("MatrixLinearOperator", Am), ("Matrix + 0.5*I (Sum)", (Am + 0.5*LinearOperator.identity(X)).with_traits(Traits.POSITIVE_DEFINITE)), ("2*Matrix (Scaled)", (2.0*Am).with_traits(Traits.POSITIVE_DEFINITE))]:
    cm = Counter(); orig = type(op)._value
    calls = [0]
    def counting_value(self, x, _orig=orig): calls[0] += 1; return _orig(self, x)
    type(op)._value = counting_value
    t = timeit(lambda: JacobiPreconditioner()(op), repeats=1)
    type(op)._value = orig
    print(f"Jacobi on {name:28s}: {calls[0]:5d} applications, {t*1e3:7.1f} ms")
# Tikhonov normal operator: Jacobi and NormalDiagonal in the model space
m, d = 2000, 300
F = rng.standard_normal((d, m))
Y = EuclideanSpace(d)
fwd = LinearOperator.from_matrix(X, Y, F, form="components")
fam = TikhonovFamily(fwd, formalism='model_space')
N = fam.at(1.0)
print(type(N).__name__, "assembled:", type(N.assembled).__name__)
calls = [0]; orig = type(fwd)._value
def cv(self, x, _o=orig): calls[0] += 1; return _o(self, x)
type(fwd)._value = cv
t = timeit(lambda: JacobiPreconditioner()(N), repeats=1)
type(fwd)._value = orig
print(f"Jacobi on TikhonovNormalOperator(model space, dim {m}): {calls[0]} forward applications, {t*1e3:.0f} ms; exact diag from F: {timeit(lambda: np.einsum('ij,ij->j', F, F))*1e3:.2f} ms")
t2 = timeit(lambda: JacobiPreconditioner(samples=20)(N), repeats=1)
print(f"Jacobi(samples=20) on same: {t2*1e3:.0f} ms")

t3 = timeit(lambda: random_diagonal(N, samples=20, form="galerkin"), repeats=1)
print(f"random_diagonal(samples=20) alone: {t3*1e3:.0f} ms")
