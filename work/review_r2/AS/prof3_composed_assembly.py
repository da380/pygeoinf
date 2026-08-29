from common import *
from scipy.linalg import lu_factor, lu_solve
from pygeoinf2 import LinearOperator, EuclideanSpace, DiagonalLinearOperator, LUSolver, CholeskySolver, Traits
n = 2000
rng = np.random.default_rng(0)
M1 = rng.standard_normal((n, n)); M2 = rng.standard_normal((n, n))
for name, X in [("euclid", EuclideanSpace(n)), ("dense-metric", dense_space(n))]:
    A = LinearOperator.from_matrix(X, X, M1, form="components")
    B = LinearOperator.from_matrix(X, X, M2, form="components")
    d = rng.uniform(1, 2, n)
    D = DiagonalLinearOperator(X, d)
    G = X._gram if name == "dense-metric" else np.eye(n)
    cases = {
        "(A@B).matrix()": (lambda: (A @ B).matrix(form="components"), lambda: M1 @ M2),
        "(A+B).matrix()": (lambda: (A + B).matrix(form="components"), lambda: M1 + M2),
        "(2*A).matrix()": (lambda: (2.0 * A).matrix(form="components"), lambda: 2.0 * M1),
        "A.adjoint.matrix()": (lambda: A.adjoint.matrix(form="components"), lambda: np.linalg.solve(G, M1.T @ G)),
        "D.matrix()": (lambda: D.matrix(form="components"), lambda: np.diag(d)),
        "(A+D).matrix()": (lambda: (A + D).matrix(form="components"), lambda: M1 + np.diag(d)),
    }
    for k, (slow, fast) in cases.items():
        r = interleave({"v2": slow, "direct": fast}, repeats=2)
        ok = np.allclose(slow(), fast(), atol=1e-8*n)
        print(f"{name:12s} {k:22s} v2 {r['v2']*1e3:7.0f} ms  direct {r['direct']*1e3:7.0f} ms  ratio {r['v2']/r['direct']:6.1f}  agree={ok}")
    # inverse matrix
    inv = LUSolver()(A)
    f = lu_factor(M1)
    r = interleave({"v2 LU(A).matrix()": lambda: inv.matrix(form="components"), "lu_solve(I)": lambda: lu_solve(f, np.eye(n))}, repeats=2)
    print(f"{name:12s} inverse.matrix()      v2 {r['v2 LU(A).matrix()']*1e3:7.0f} ms  direct {r['lu_solve(I)']*1e3:7.0f} ms")
    # Tikhonov-shaped direct solve: Cholesky on A.adjoint @ A + t I where A stored
    S = M1.T @ M1 + n*np.eye(n)
    As = LinearOperator.from_matrix(X, X, G @ S if name=="dense-metric" else S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
    # NB: for dense metric the galerkin matrix must be symmetric: G @ (G^-1 S) = S. Use S directly:
    As = LinearOperator.from_matrix(X, X, S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
    N = (As + 0.5 * LinearOperator.identity(X)).with_traits(Traits.POSITIVE_DEFINITE)
    print(type(N).__name__, [type(t).__name__ for t in getattr(N, "terms", [])])
    r = interleave({"Chol(As + 0.5 I)": lambda: CholeskySolver()(N), "Chol(As)": lambda: CholeskySolver()(As)}, repeats=2)
    print(f"{name:12s} Cholesky: sum node {r['Chol(As + 0.5 I)']*1e3:7.0f} ms  stored {r['Chol(As)']*1e3:7.0f} ms")
