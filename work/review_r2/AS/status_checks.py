from common import *
import warnings
from pygeoinf2 import LinearOperator, EuclideanSpace, DiagonalLinearOperator, Traits, DirectSum, AffineOperator, MatrixLinearOperator
from pygeoinf2.algebra.linearisation import Linearisation
from pygeoinf2.testing import check_traits, check_operator
rng = np.random.default_rng(0)
def check(label, fn):
    try:
        out = fn(); print(f"[{label}] -> {out}")
    except Exception as e:
        print(f"[{label}] RAISES {type(e).__name__}: {str(e)[:90]}")
X = EuclideanSpace(5); Y = EuclideanSpace(3)
M = rng.standard_normal((3, 5))
wide_no_adj = LinearOperator.from_callables(X, Y, lambda x: M @ x)
check("A Should-7: matrix(by=auto) on wide op without adjoint", lambda: wide_no_adj.matrix().shape)
D3 = make_dense_metric_space(3)
Dd = DiagonalLinearOperator(D3, np.array([1.0, 2.0, 3.0]))
check("A Must-4: DiagonalLinearOperator.sqrt on dense metric (d>0)", lambda: Dd.sqrt.eigenvalues)
check("A Must-4: D.log_determinant on dense metric", lambda: Dd.log_determinant)
Dt = DiagonalLinearOperator(D3, np.array([1.0, 2.0, 3.0]), traits=Traits.SELF_ADJOINT)
check("A Must-4: _rebuild keeps user traits (2*D).traits", lambda: (2.0*Dt).traits)
De = DiagonalLinearOperator(X, np.ones(5))
check("A Consider-16: D + I stays diagonal", lambda: type(De + LinearOperator.identity(X)).__name__)
check("A Consider-16: D + 2*I stays diagonal", lambda: type(De + 2.0*LinearOperator.identity(X)).__name__)
A = LinearOperator.from_matrix(X, X, np.eye(5), form="components")
aff = AffineOperator(A, np.ones(5))
check("A Consider-17: Affine @ Affine", lambda: type(aff @ aff).__name__)
lin = Linearisation(np.ones(2), np.ones(2), A)
check("A Should-11: Linearisation == Linearisation", lambda: lin == lin)
check("A Should-11: hash(Linearisation)", lambda: hash(lin))
check("A Should-11: eigenvalues is internal array", lambda: De.eigenvalues is De._eigenvalues)
from pygeoinf2.algebra.spaces import DiagonalMetricSpace
W = weighted_space(5)
check("A Should-11: metric_values is internal array", lambda: W.metric_values is W._metric_values)
check("A Consider-15: DiagonalMetricSpace.gram_matrix overridden", lambda: type(W).gram_matrix is DiagonalMetricSpace.gram_matrix or "inherited from CoordinateSpace (column loop)")
import pygeoinf2.algebra.spaces as sp
print("  gram_matrix owner:", [k for k in ("CoordinateSpace","DiagonalMetricSpace") if "gram_matrix" in getattr(sp, k).__dict__])
# _Zero blocks in BlockOperator: count zero() and axpy calls
from pygeoinf2 import BlockLinearOperator
c_zero = Counter(); c_axpy = Counter()
Z = EuclideanSpace(4)
Z.zero = c_zero.wrap(Z.zero); Z.axpy = c_axpy.wrap(Z.axpy)
I = LinearOperator.identity(Z); O = LinearOperator.zero(Z)
B = BlockLinearOperator([[I, O], [O, I]])
B((np.ones(4), np.ones(4)))
print(f"[A Consider-16 _Zero blocks] BlockLinearOperator 2x2 with two zero blocks: {c_zero.n} zero() and {c_axpy.n} axpy per application")
# from_vectors meaning
V = LinearOperator.from_vectors(X, [np.eye(5)[0], np.eye(5)[1]])
print("[A Should-9] from_vectors maps", V.domain, "->", V.codomain)
# with_traits preserves class
check("A Must-4: with_traits preserves class", lambda: type(A.with_traits(Traits.SELF_ADJOINT)).__name__)
# BiCGStab breakdown reporting
from pygeoinf2 import BiCGStabSolver, MinResSolver, GMRESSolver, FlexibleCGSolver, CGSolver, LUSolver
import pygeoinf2
print("[S Must-1] exported:", all(hasattr(pygeoinf2, n) for n in ("FlexibleCGSolver", "GMRESSolver")))
from pygeoinf2.numerics.solvers import SolverLike
print("[S Consider-19] SolverLike is", type(SolverLike).__name__)
# InvariantDistance metric on DiagonalMetricSpace data space: covered by test? 
import subprocess
out = subprocess.run(["grep", "-rn", "InvariantDistance", "/home/david/dev/pygeoinf/pygeoinf2/tests/", "-l"], capture_output=True, text=True).stdout
print("[S Should-10] tests mentioning InvariantDistance:", out.split())
# assembled() form
print("[A Must-3 assembled form]", A.assembled().stored_form, "(MatrixLinearOperator returns self);", LinearOperator.from_callables(X, X, lambda x: x, adjoint=lambda x: x).assembled().stored_form)
# DampedSolves: factorisations per solve at the same multiplier
from pygeoinf2.numerics.root_find import DampedSolves
import pygeoinf2.numerics.solvers as S
cnt = Counter(); orig = S.cho_factor; S.cho_factor = cnt.wrap(orig)
from pygeoinf2 import CholeskySolver
Spd = LinearOperator.from_matrix(X, X, np.eye(5)*3, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
ds = DampedSolves(Spd, LinearOperator.identity(X), CholeskySolver(), traits=Traits.POSITIVE_DEFINITE)
ds.solve(1.0, np.ones(5)); ds.solve(1.0, np.ones(5)); ds.solve(1.0, np.ones(5))
print(f"[S Consider-23] DampedSolves: 3 solves at the same multiplier -> {cnt.n} Cholesky factorisations")
S.cho_factor = orig
# Woodbury Q applications per application
from pygeoinf2 import WoodburyPreconditioner
Xm, Yd = EuclideanSpace(6), EuclideanSpace(4)
F = LinearOperator.from_matrix(Xm, Yd, rng.standard_normal((4, 6)), form="components")
cq = Counter()
Q = LinearOperator.self_adjoint(Xm, cq.wrap(lambda x: 2.0*x), traits=Traits.POSITIVE_DEFINITE)
Rn = LinearOperator.self_adjoint(Yd, lambda x: x, traits=Traits.POSITIVE_DEFINITE)
Wp = WoodburyPreconditioner(F, Q, Rn, solver=LUSolver())
mf = Wp.model_form(); cq.n = 0; mf(np.ones(6)); print(f"[S Consider-20] Woodbury model_form: {cq.n} applications of Q per application")
# IdentityPreconditioner returns input
from pygeoinf2 import IdentityPreconditioner
y = np.ones(5); print("[S Consider-25] IdentityPreconditioner returns the input object:", IdentityPreconditioner()(A)(y) is y)
# JacobiPreconditioner docstring claim on Sum
